import os, random, math

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import sys
import importlib
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timedelta
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
import cv2
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from einops import rearrange
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

DMD_ROOT = os.environ.get("DMD_ROOT")
if not DMD_ROOT:
    raise RuntimeError("Set DMD_ROOT to the local dmd_diffusion directory")
DMD_SRC_PATH = os.path.join(DMD_ROOT, "src")

_conflict_prefixes = ("utils", "models", "configs", "data", "datasets", "scripts")
_saved_modules = {}
for k in list(sys.modules.keys()):
    if k.split(".")[0] in _conflict_prefixes:
        _saved_modules[k] = sys.modules.pop(k)

_old_cwd = os.getcwd()
_old_sys_path = sys.path.copy()

os.chdir(DMD_SRC_PATH)
sys.path.insert(0, DMD_SRC_PATH)
sys.path_importer_cache.clear()
importlib.invalidate_caches()

try:
    from utils.camera_tools import rel_camera_ray_encoding, freq_enc
    from utils.score_tools import Score_sde_model, Score_modifier
except ImportError:
    from utils import rel_camera_ray_encoding, freq_enc
    from utils import Score_sde_model, Score_modifier

from models.score_sde import sde_lib
from models.score_sde.ncsnpp_dual import NCSNpp_dual
from models.score_sde.configs.LDM import LDM_config
from models.vqgan.vqgan import VQModel
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
from models.score_sde.layerspp import ResnetBlockBigGANpp
import functools

for k in list(sys.modules.keys()):
    if k.split(".")[0] in _conflict_prefixes:
        del sys.modules[k]

os.chdir(_old_cwd)
sys.path = _old_sys_path
sys.modules.update(_saved_modules)
sys.path_importer_cache.clear()
importlib.invalidate_caches()
# ==============================================================================================


from utils.model_utils import (
    load_condition_models,
    load_latent_models,
    load_vae_models,
    load_diffusion_model,
    count_model_parameters,
    unwrap_model,
    forward_pass,
)
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import (
    get_latents,
    get_text_conditions,
    gen_noise_from_condition_frame_latent,
    randn_tensor,
    apply_color_jitter_to_video,
)
from utils.extra_utils import act_metric
from utils.get_my_batch_traj import get_batch_arm_traj_maps, simple_radius_gen_func
from utils.get_my_batch_ray import get_ray_maps_batch
from utils.get_ray_maps import get_ray_maps
from utils.geometry_utils import resize_traj_and_ray

from data.dataset import LerobotDataFormatter

LOG_LEVEL = "INFO"
logger = get_logger("gesim_runner")
logger.setLevel(LOG_LEVEL)

NUM_FOR_A_VIDEO = 8
TOTAL_REPEATS = 10
TASK_NAME = "pushbag_m2_4"
DEMO_NUM = 5
PRE_CONVERSION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def tensor_video_to_cv2_frames(
    video_tensor: torch.Tensor, target_size=(480, 640)
) -> List[np.ndarray]:
    video_np = video_tensor.permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()

    video_np = (video_np + 1) / 2 * 255
    video_np = video_np.astype(np.uint8)

    frames = []
    for frame in video_np:
        frame_resized = cv2.resize(
            frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
        )
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return frames


def generate_dummy_depth_frames(num_frames: int, size=(480, 640)) -> List[np.ndarray]:
    depth_frames = []
    for _ in range(num_frames):
        depth_img = np.random.uniform(0, 1000, size).astype(np.float32)
        depth_frames.append(depth_img)
    return depth_frames


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_frame_as_image(frame: torch.Tensor, save_path: str):
    frame_np = frame.permute(1, 2, 0).to(torch.float32).cpu().numpy()
    frame_np = (frame_np + 1) / 2 * 255
    frame_np = frame_np.astype(np.uint8)
    img = Image.fromarray(frame_np)
    img.save(save_path)
    print(f"帧图片已保存至: {save_path}")


def extract_frame_features(frame: torch.Tensor, target_size=(64, 64)):
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)

    frame_down = F.interpolate(
        frame.float(), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0)

    features = frame_down.flatten().to(torch.float32).cpu().numpy()
    return features


def visualize_tsne_for_frames(
    gen_frames, real_frames, repeat_num, save_path="tsne_gen_last_vs_real_first.png"
):
    print("Extracting features from frames...")
    gen_features = []
    real_features = []
    for gen_frame, real_frame in zip(gen_frames, real_frames):
        gen_feat = extract_frame_features(gen_frame)
        real_feat = extract_frame_features(real_frame)
        gen_features.append(gen_feat)
        real_features.append(real_feat)

    gen_features_np = np.array(gen_features)  # [repeat_num, feature_dim]
    real_features_np = np.array(real_features)  # [repeat_num, feature_dim]
    all_features = np.vstack(
        [gen_features_np, real_features_np]
    )  # [2*repeat_num, feature_dim]

    print("Calculating distances between generated and real frames...")
    distances = []
    for gen_feat, real_feat in zip(gen_features, real_features):
        dist = np.linalg.norm(gen_feat - real_feat)
        distances.append(dist)
    distances_np = np.array(distances)

    closest_idx = np.argmin(distances_np)
    farthest_idx = np.argmax(distances_np)
    closest_dist = distances_np[closest_idx]
    farthest_dist = distances_np[farthest_idx]
    print(f"\nClosest frame index: {closest_idx}, distance: {closest_dist:.4f}")
    print(f"Farthest frame index: {farthest_idx}, distance: {farthest_dist:.4f}")

    indexed_distances = list(enumerate(distances_np))
    indexed_distances.sort(key=lambda x: x[1])
    top_half = repeat_num // 2
    top_half_indices = [idx for idx, _ in indexed_distances[:top_half]]
    print(f"筛选出前 {top_half} 个最近轨迹，索引: {top_half_indices[:5]}...")

    print("Running TSNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, repeat_num - 1),
        random_state=42,
        learning_rate=150,
        init="pca",
    )
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(12, 8))
    plt.scatter(
        tsne_results[:repeat_num, 0],
        tsne_results[:repeat_num, 1],
        c="#2E86AB",
        label="Generated Video Last Frame",
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidth=0.5,
    )
    plt.scatter(
        tsne_results[repeat_num:, 0],
        tsne_results[repeat_num:, 1],
        c="#A23B72",
        label="Original Future First Frame",
        alpha=0.7,
        s=60,
        marker="^",
        edgecolors="white",
        linewidth=0.5,
    )

    plt.title(
        "TSNE Comparison: Generated Last Frame vs Original First Frame",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("TSNE Dimension 1", fontsize=14)
    plt.ylabel("TSNE Dimension 2", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.2, linestyle="--")

    gen_centroid = np.mean(tsne_results[:repeat_num], axis=0)
    real_centroid = np.mean(tsne_results[repeat_num:], axis=0)
    centroid_dist = np.linalg.norm(gen_centroid - real_centroid)
    gen_std = np.mean(np.std(tsne_results[:repeat_num], axis=0))
    real_std = np.mean(np.std(tsne_results[repeat_num:], axis=0))

    stats_text = f"""
    Centroid Distance: {centroid_dist:.4f}
    Closest Frame Distance: {closest_dist:.4f}
    Farthest Frame Distance: {farthest_dist:.4f}
    Generated Std: {gen_std:.4f}
    Real Std: {real_std:.4f}
    Top Half Selected: {top_half}
    """
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        verticalalignment="top",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"TSNE plot saved to: {save_path}")

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    closest_gen_frame = gen_frames[closest_idx].to(torch.float32)
    farthest_gen_frame = gen_frames[farthest_idx].to(torch.float32)
    real_frame = real_frames[0].to(torch.float32)

    closest_gen_path = os.path.join(save_dir, "closest_generated_frame.png")
    farthest_gen_path = os.path.join(save_dir, "farthest_generated_frame.png")
    real_frame_path = os.path.join(save_dir, "real_reference_frame.png")

    save_frame_as_image(closest_gen_frame, closest_gen_path)
    save_frame_as_image(farthest_gen_frame, farthest_gen_path)
    save_frame_as_image(real_frame, real_frame_path)

    metrics = {
        "centroid_distance": centroid_dist,
        "generated_std": gen_std,
        "real_std": real_std,
        "generated_centroid": gen_centroid,
        "real_centroid": real_centroid,
        "closest_index": closest_idx,
        "closest_distance": closest_dist,
        "farthest_index": farthest_idx,
        "farthest_distance": farthest_dist,
        "all_distances": distances_np.tolist(),
        "top_half_indices": top_half_indices,
    }

    # metrics_path = os.path.join(save_dir, "tsne_metrics.json")
    # with open(metrics_path, "w") as f:
    #     json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}, f, indent=4)
    # print(f"TSNE metrics saved to: {metrics_path}")

    print("\n===== TSNE Analysis Metrics =====")
    print(f"Distance between centroids: {centroid_dist:.4f}")
    print(f"Closest frame distance: {closest_dist:.4f}")
    print(f"Farthest frame distance: {farthest_dist:.4f}")
    print(f"Generated frames std: {gen_std:.4f}")
    print(f"Real frames std: {real_std:.4f}")
    print(f"Top half indices: {top_half_indices[:5]}...")

    return tsne_results, metrics, top_half_indices


def actions_to_homogeneous_matrices(actions_tensor):
    assert (
        len(actions_tensor.shape) == 3 and actions_tensor.shape[-1] == 8
    ), f"输入张量需为 [b, t, 8]，当前形状：{actions_tensor.shape}"

    b, t, _ = actions_tensor.shape
    device = actions_tensor.device
    dtype = actions_tensor.dtype

    translation = actions_tensor[..., :3]  # [b, t, 3]
    quaternion_wxyz = actions_tensor[..., 3:7]  # [b, t, 4]
    quaternion_wxyz = F.normalize(quaternion_wxyz, p=2, dim=-1)
    qw, qx, qy, qz = torch.unbind(quaternion_wxyz, dim=-1)

    r11 = 1 - 2 * qy**2 - 2 * qz**2
    r12 = 2 * qx * qy - 2 * qz * qw
    r13 = 2 * qx * qz + 2 * qy * qw

    r21 = 2 * qx * qy + 2 * qz * qw
    r22 = 1 - 2 * qx**2 - 2 * qz**2
    r23 = 2 * qy * qz - 2 * qx * qw

    r31 = 2 * qx * qz - 2 * qy * qw
    r32 = 2 * qy * qz + 2 * qx * qw
    r33 = 1 - 2 * qx**2 - 2 * qy**2

    rot_matrix = torch.stack(
        [
            torch.stack([r11, r12, r13], dim=-1),
            torch.stack([r21, r22, r23], dim=-1),
            torch.stack([r31, r32, r33], dim=-1),
        ],
        dim=-2,
    )

    homo_matrices = (
        torch.eye(4, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(b, t, 1, 1)
    )
    homo_matrices[..., :3, :3] = rot_matrix
    homo_matrices[..., :3, 3] = translation
    homo_matrices = homo_matrices.unsqueeze(1)
    return homo_matrices


def gripper_state_to_tensor(actions_tensor, n_view=1, H=384, W=512):
    gripper_state = actions_tensor[..., 7]  # [b, t]
    gripper_tensor = gripper_state.unsqueeze(1)  # [b, 1, t]
    gripper_tensor = gripper_tensor.unsqueeze(2)  # [b, 1, n_view, t]
    gripper_tensor = gripper_tensor.unsqueeze(-1)  # [b, 1, n_view, t, H]
    gripper_tensor = gripper_tensor.unsqueeze(-1)  # [b, 1, n_view, t, H, W]
    gripper_tensor = gripper_tensor.repeat(1, 1, 1, 1, H, W)
    return gripper_tensor


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None


class GeSimTrainer:

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:

        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print("You are not loading the pretrained weights, please check the code.")
        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.vae = None
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(
                    self.args.output_dir, self.args.sub_folder
                )
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, "config.json"), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)

            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor(
                [len(save_folder_bytes)], device=self.state.accelerator.device
            )
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(
                self.state.accelerator.device
            )
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(
                folder_len_tensor.item(),
                dtype=torch.uint8,
                device=self.state.accelerator.device,
            )
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        init_logging(self.save_folder, rank=self.state.accelerator.process_index)
        self.lerobot_formatter = LerobotDataFormatter(
            output_dir=f"/mnt/workspace/data/dmd_augment_{DEMO_NUM}_{NUM_FOR_A_VIDEO}_{TOTAL_REPEATS}_lerobot_demo_{TASK_NAME}",
            task_name="robot_manipulation",
            chunk_size=10000,
            action_dim=8,
        )
        self.lerobot_episode_counter = 0

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = (
            "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        )
        report_to = (
            None if self.args.report_to.lower() == "none" else self.args.report_to
        )

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed, gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.state.weight_dtype = weight_dtype

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized GeSimTrainer")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir

    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")
        local_rank = int(os.environ["LOCAL_RANK"])

        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(
            **self.args.data["train"],
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            multiprocessing_context=None,
        )
        logger.info(
            f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n"
        )

        if "val" in self.args.data:
            self.prepare_val_dataset()

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(
            **self.args.data["val"],
        )

        subset = self.val_dataset

        self.val_dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=getattr(self.args, "val_shuffle", False),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        logger.info(
            f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<"
        )
        logger.info(
            f">>>>>>>>>>>>>Val Dataloader Batch Num: {len(self.val_dataloader)}<<<<<<<<<<<<<<<<<<"
        )

    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class,
            getattr(self.args, "tokenizer_class_path", "transformers"),
        )
        textenc_class = import_custom_class(
            self.args.textenc_class,
            getattr(self.args, "textenc_class_path", "transformers"),
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            (
                self.args.pretrained_model_name_or_path
                if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path")
                else self.args.tokenizer_pretrained_model_name_or_path
            ),
            load_weights=self.args.load_weights,
        )
        self.tokenizer, text_encoder = (
            cond_models["tokenizer"],
            cond_models["text_encoder"],
        )
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(
            self.tokenizer, self.text_encoder, prompt=""
        )
        self.uncond_prompt_embeds = self.text_uncond["prompt_embeds"]
        self.uncond_prompt_attention_mask = self.text_uncond["prompt_attention_mask"]

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, "vae_path", False):
            self.vae = (
                load_vae_models(vae_class, self.args.vae_path)
                .to(device, dtype=dtype)
                .eval()
            )
        else:
            self.vae = (
                load_latent_models(vae_class, self.args.pretrained_model_name_or_path)[
                    "vae"
                ]
                .to(device, dtype=dtype)
                .eval()
            )
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f"SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}")
        logger.info(f"TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}")

        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class,
            getattr(self.args, "diffusion_model_class_path", "transformers"),
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights
            and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f"Total parameters for transformer model:{total_params}")

        ### Load Diffuser Scheduler
        args = self.args
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class,
            getattr(self.args, "diffusion_scheduler_class_path", "diffusers"),
        )
        if hasattr(diffusion_scheduler_class, "from_pretrained") and os.path.exists(
            os.path.join(args.pretrained_model_name_or_path, "scheduler")
        ):
            self.scheduler = diffusion_scheduler_class.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, "scheduler")
            )
        else:
            if hasattr(args, "diffusion_scheduler_args"):
                self.scheduler = diffusion_scheduler_class(
                    **args.diffusion_scheduler_args
                )
            else:
                self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class,
            getattr(self.args, "pipeline_class_path", "diffusers"),
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        components_to_disable_grads = [self.text_encoder, self.vae]

        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if (
            torch.backends.mps.is_available()
            and self.state.weight_dtype == torch.bfloat16
        ):
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        diffusion_model_trainable_params = []
        if train_mode == "action_only":
            for name, param in self.diffusion_model.named_parameters():
                if "action_" in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "video_only":
            for name, param in self.diffusion_model.named_parameters():
                if "action_" not in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "all" or train_mode == "action_full":
            for name, param in self.diffusion_model.named_parameters():
                param.requires_grad = True
                diffusion_model_trainable_params.append(param)
        else:
            raise NotImplementedError

        num_trainable_params = sum(p.numel() for p in diffusion_model_trainable_params)
        logger.info(f"Total trainable parameters: {num_trainable_params}")

        diffusion_model_parameters_with_lr = {
            "params": diffusion_model_trainable_params,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [diffusion_model_parameters_with_lr]
        self.state.num_trainable_parameters = sum(
            p.numel() for p in diffusion_model_trainable_params
        )

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit=self.args.optimizer_8bit,
            use_torchao=self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.state.train_steps is None:
            self.state.train_steps = (
                self.state.train_epochs * num_update_steps_per_epoch
            )
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps
            * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self):
        (
            self.diffusion_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.state.accelerator.prepare(
            self.diffusion_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)

    def expand_to_bvt(self, original_array, b, v, t):
        expanded = np.expand_dims(original_array, axis=(0, 1, 2))
        result = np.repeat(expanded, repeats=b, axis=0)
        result = np.repeat(result, repeats=v, axis=1)
        result = np.repeat(result, repeats=t, axis=2)
        return result

    def generate_virtual_trajectory(
        self,
        init_pose: torch.Tensor,
        end_pose: torch.Tensor,
        offset_scale: float = 0.1,
        num_points: int = 25,
    ) -> torch.Tensor:
        assert (
            init_pose.shape[-1] == 8
        ), f"初始姿态必须为8维，当前形状: {init_pose.shape}"
        assert end_pose.shape[-1] == 8, f"结束姿态必须为8维，当前形状: {end_pose.shape}"
        assert num_points == 25, "固定生成25个轨迹点，请勿修改"

        device = init_pose.device
        dtype = init_pose.dtype

        init_trans = init_pose[:3]
        init_quat = init_pose[3:7]
        init_gripper = init_pose[7:8]

        end_trans = end_pose[:3]

        vec_init2end = end_trans - init_trans
        norm_init2end = torch.norm(vec_init2end, p=2)

        max_retries = 100
        best_offset_trans = None

        for _ in range(max_retries):
            rand_vec = torch.randn(3, device=device, dtype=dtype)
            offset_vec = (
                torch.nn.functional.normalize(rand_vec, p=2, dim=0) * offset_scale
            )

            candidate_trans = init_trans + offset_vec

            vec_init2candidate = candidate_trans - init_trans
            norm_init2candidate = torch.norm(vec_init2candidate, p=2)

            if norm_init2end < 1e-8:
                best_offset_trans = candidate_trans
                break

            if norm_init2candidate < 1e-8:
                continue

            vec_init2end_norm = vec_init2end / norm_init2end
            vec_init2candidate_norm = vec_init2candidate / norm_init2candidate

            dot_product = torch.clamp(
                torch.dot(vec_init2end_norm, vec_init2candidate_norm), -1.0, 1.0
            )
            angle_rad = torch.acos(dot_product)
            angle_deg = torch.rad2deg(angle_rad)

            if angle_deg > 120.0:
                best_offset_trans = candidate_trans
                break

        if best_offset_trans is None:
            best_offset_trans = init_trans + offset_vec

        offset_pose = torch.cat([best_offset_trans, init_quat, init_gripper], dim=-1)

        interp_points_out = torch.linspace(0, 1, steps=12, device=device, dtype=dtype)
        traj_out = []
        for alpha in interp_points_out:
            pose = init_pose * (1 - alpha) + offset_pose * alpha
            traj_out.append(pose)
        traj_out = torch.stack(traj_out)

        interp_points_in = torch.linspace(0, 1, steps=12, device=device, dtype=dtype)
        traj_in = []
        for alpha in interp_points_in:
            pose = offset_pose * (1 - alpha) + init_pose * alpha
            traj_in.append(pose)
        traj_in = torch.stack(traj_in)

        virtual_actions = torch.cat(
            [traj_out, offset_pose.unsqueeze(0), traj_in], dim=0
        )

        return virtual_actions

    # def generate_virtual_trajectory(self, init_pose: torch.Tensor, end_pose: torch.Tensor, offset_scale: float = 0.1, num_points: int = 25) -> torch.Tensor:
    #     """
    #     """

    #     device = init_pose.device
    #     dtype = init_pose.dtype

    #     init_trans = init_pose[:3]
    #     init_quat = init_pose[3:7]
    #     init_gripper = init_pose[7:8]

    #     end_trans = end_pose[:3]

    #     dist_init_to_end = torch.norm(init_trans - end_trans, p=2)

    #     best_offset_trans = None

    #     for _ in range(max_retries):
    #         rand_vec = torch.randn(3, device=device, dtype=dtype)
    #         offset_vec = torch.nn.functional.normalize(rand_vec, p=2, dim=0) * offset_scale

    #         candidate_trans = init_trans + offset_vec

    #         dist_candidate_to_end = torch.norm(candidate_trans - end_trans, p=2)

    #         if dist_candidate_to_end >= dist_init_to_end:
    #             best_offset_trans = candidate_trans
    #             break

    #     if best_offset_trans is None:
    #         best_offset_trans = init_trans + offset_vec

    #     offset_pose = torch.cat([best_offset_trans, init_quat, init_gripper], dim=-1)

    #     interp_points_out = torch.linspace(0, 1, steps=12, device=device, dtype=dtype)
    #     traj_out = []
    #     for alpha in interp_points_out:
    #         # (1-a)*Init + a*Offset
    #         pose = init_pose * (1 - alpha) + offset_pose * alpha
    #         traj_out.append(pose)
    #     traj_out = torch.stack(traj_out)

    #     interp_points_in = torch.linspace(0, 1, steps=12, device=device, dtype=dtype)
    #     traj_in = []
    #     for alpha in interp_points_in:
    #         # (1-a)*Offset + a*Init
    #         pose = offset_pose * (1 - alpha) + init_pose * alpha
    #         traj_in.append(pose)
    #     traj_in = torch.stack(traj_in)

    #     virtual_actions = torch.cat([traj_out, offset_pose.unsqueeze(0), traj_in], dim=0)

    #     return virtual_actions

    def augment_dataset(
        self, save_root: str, num_samples: int = 50, offset_scale: float = 0.35
    ):
        from utils import save_video

        accelerator = self.state.accelerator
        device = accelerator.device
        weight_dtype = self.state.weight_dtype
        os.makedirs(save_root, exist_ok=True)

        def cleanup_distributed():
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                    accelerator.print("分布式进程组已关闭")
                except Exception as e:
                    accelerator.print(f"关闭分布式进程组警告: {e}")

        accelerator.print(">>> 开始加载 DMD 模型...")
        dmd_ckpt_path = os.path.join(
            DMD_ROOT, "training_output", "checkpoints", "10000-00000000.pth"
        )
        dmd_steps = 250
        dmd_batch_size = 64

        vqgan = VQModel(**vqgan_32_4_config).to(device)
        vqgan.eval()

        dmd_config = LDM_config()
        score_model = NCSNpp_dual(dmd_config).to(device)
        sde = sde_lib.VESDE(
            sigma_min=dmd_config.model.sigma_min,
            sigma_max=dmd_config.model.sigma_max,
            N=dmd_steps,
        )

        ResnetBlock = functools.partial(
            ResnetBlockBigGANpp,
            act=torch.nn.SiLU(),
            dropout=False,
            fir=True,
            fir_kernel=[1, 3, 3, 1],
            init_scale=0,
            skip_rescale=True,
            temb_dim=None,
        )
        ray_downsampler = torch.nn.Sequential(
            ResnetBlock(in_ch=56, out_ch=128, down=True),
            ResnetBlock(in_ch=128, out_ch=128, down=True),
        ).to(device)

        score_sde = Score_sde_model(
            score_model,
            sde,
            ray_downsampler,
            rays_require_downsample=False,
            rays_as_list=True,
        )

        checkpoint = torch.load(dmd_ckpt_path, map_location=device, weights_only=False)
        adapted_state = {}
        for k, v in checkpoint["score_sde_model"].items():
            key_parts = k.split(".")
            if key_parts[1] == "module":
                key_parts.pop(1)
            new_key = ".".join(key_parts)
            adapted_state[new_key] = v
        score_sde.load_state_dict(adapted_state)

        ema = ExponentialMovingAverage(score_sde.parameters(), decay=0.999)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(score_sde.parameters())

        modifier = Score_modifier(score_sde.score_model, max_batch_size=dmd_batch_size)
        score_sde.score_model = modifier
        score_sde.eval()
        accelerator.print(">>> DMD 模型加载完成！")
        # ==========================================================

        mem_size = self.args.data["train"]["n_previous"]
        future_size = self.args.data["train"]["chunk"]
        h, w = 384, 512
        v = 1

        val_dataloader = self.val_dataloader
        sample_count = 0

        P_tensor = torch.tensor(PRE_CONVERSION, dtype=weight_dtype, device=device)

        P_tensor_high_prec = P_tensor.to(dtype=torch.float32)
        P_inv_tensor_high_prec = torch.inverse(P_tensor_high_prec)
        P_inv_tensor = P_inv_tensor_high_prec.to(dtype=weight_dtype)

        with torch.no_grad():
            num_for_a_video = NUM_FOR_A_VIDEO
            for index_num_for_a_video in range(num_for_a_video):
                demo_idx_count = 0
                for batch_idx, batch in enumerate(val_dataloader):
                    if demo_idx_count >= DEMO_NUM:
                        break

                    try:
                        raw_video = batch["video"].to(device, dtype=weight_dtype)
                        raw_actions = batch["actions"].to(device, dtype=weight_dtype)
                        captions = batch["caption"]
                        b_size = raw_video.shape[0]

                        for idx in range(b_size):
                            if demo_idx_count >= DEMO_NUM:
                                break
                            demo_idx_count += 1
                            sample_count += 1

                            sample_id = f"aug_sample_{sample_count:04d}"
                            sample_save_dir = os.path.join(save_root, sample_id)
                            os.makedirs(sample_save_dir, exist_ok=True)
                            gen_last_frames = []
                            gen_videos_all = []
                            gen_actions_all = []
                            total_repeats = TOTAL_REPEATS
                            batch_size = 20
                            num_batches = (total_repeats + batch_size - 1) // batch_size

                            init_video_single = raw_video[
                                idx : idx + 1, :, :, :mem_size
                            ]
                            init_action_single = raw_actions[idx, mem_size]
                            end_action_single = raw_actions[idx, mem_size + 5]
                            original_future_video_single = raw_video[
                                idx : idx + 1, :, :, mem_size : mem_size + 25
                            ]
                            original_future = rearrange(
                                original_future_video_single, "1 c () t h w -> c t h w"
                            ).cpu()
                            original_future_actions = raw_actions[
                                idx, mem_size : mem_size + 25
                            ].to(torch.float32)

                            for batch_idx_sub in range(num_batches):
                                start_idx = batch_idx_sub * batch_size
                                end_idx = min(
                                    (batch_idx_sub + 1) * batch_size, total_repeats
                                )
                                current_batch_size = end_idx - start_idx
                                if current_batch_size == 0:
                                    break

                                virtual_actions_list = []
                                init_video_batch_list = []

                                for repeat_in_batch in range(current_batch_size):
                                    virtual_action = self.generate_virtual_trajectory(
                                        init_pose=init_action_single,
                                        end_pose=end_action_single,
                                        offset_scale=offset_scale,
                                        num_points=25,
                                    )
                                    virtual_actions_list.append(virtual_action)
                                    init_video_batch_list.append(init_video_single)

                                virtual_actions_batch = torch.stack(
                                    virtual_actions_list, dim=0
                                )  # [B, 25, 8]
                                init_video_batch = torch.cat(
                                    init_video_batch_list, dim=0
                                )  # [B, C, V, T_mem, H, W]

                                poses_4x4 = actions_to_homogeneous_matrices(
                                    virtual_actions_batch
                                ).squeeze(1)

                                curr_frames = init_video_batch[:, :, 0, -1, :, :]
                                generated_frames_seq = [curr_frames]

                                P_batch = P_tensor.unsqueeze(0).expand(
                                    current_batch_size, -1, -1
                                )
                                P_inv_batch = P_inv_tensor.unsqueeze(0).expand(
                                    current_batch_size, -1, -1
                                )

                                for t in range(1, 25):
                                    M_prev = poses_4x4[:, t - 1, :, :]
                                    M_curr = poses_4x4[:, t, :, :]

                                    M_prev_high_prec = M_prev.to(dtype=torch.float32)
                                    M_prev_inv_high_prec = torch.linalg.inv(
                                        M_prev_high_prec
                                    )
                                    M_prev_inv = M_prev_inv_high_prec.to(
                                        dtype=poses_4x4.dtype
                                    )

                                    T_rel = torch.bmm(M_prev_inv, M_curr)  # [B, 4, 4]
                                    T_dmd = torch.bmm(
                                        P_inv_batch, torch.bmm(T_rel, P_batch)
                                    )  # [B, 4, 4]

                                    curr_frames_256 = F.interpolate(
                                        curr_frames,
                                        size=(256, 256),
                                        mode="bilinear",
                                        align_corners=False,
                                    )

                                    conditioning_ims = []
                                    ff_refs = []
                                    ff_as = []
                                    ff_bs = []

                                    focal_y = 796.8544625226342 / 720.0

                                    for b in range(current_batch_size):
                                        # encoded_im = vqgan.encode(curr_frames_256[b:b+1])
                                        input_for_vqgan = curr_frames_256[b : b + 1].to(
                                            dtype=torch.float32
                                        )
                                        encoded_im = vqgan.encode(input_for_vqgan)
                                        conditioning_ims.append([encoded_im])

                                        # tform_a = T_dmd[b].cpu().numpy()
                                        tform_a = (
                                            T_dmd[b]
                                            .to(dtype=torch.float32)
                                            .cpu()
                                            .numpy()
                                        )
                                        tform_b = np.linalg.inv(tform_a)
                                        tform_ref = np.eye(4)

                                        cam_ref = torch.Tensor(
                                            rel_camera_ray_encoding(
                                                tform_ref, 128, focal_y
                                            )
                                        ).unsqueeze(0)
                                        cam_a = torch.Tensor(
                                            rel_camera_ray_encoding(
                                                tform_a, 128, focal_y
                                            )
                                        ).unsqueeze(0)
                                        cam_b = torch.Tensor(
                                            rel_camera_ray_encoding(
                                                tform_b, 128, focal_y
                                            )
                                        ).unsqueeze(0)

                                        f_ref = F.pad(
                                            freq_enc(cam_ref), [0, 0, 0, 0, 1, 1, 0, 0]
                                        ).to(device)
                                        f_a = F.pad(
                                            freq_enc(cam_a), [0, 0, 0, 0, 1, 1, 0, 0]
                                        ).to(device)
                                        f_b = F.pad(
                                            freq_enc(cam_b), [0, 0, 0, 0, 1, 1, 0, 0]
                                        ).to(device)

                                        ff_refs.append([ray_downsampler(f_ref)])
                                        ff_as.append([ray_downsampler(f_a)])
                                        ff_bs.append([ray_downsampler(f_b)])

                                    sampling_shape = (current_batch_size, 4, 32, 32)
                                    sampling_eps = 1e-5
                                    x_latent = sde.prior_sampling(sampling_shape).to(
                                        device
                                    )
                                    timesteps = torch.linspace(
                                        sde.T, sampling_eps, sde.N, device=device
                                    )

                                    for i in tqdm(range(sde.N)):
                                        t_step = timesteps[i]
                                        vec_t = (
                                            torch.ones(
                                                current_batch_size, device=device
                                            )
                                            * t_step
                                        )
                                        x_latent, x_mean = (
                                            score_sde.reverse_diffusion_predictor(
                                                x_latent,
                                                conditioning_ims,
                                                vec_t,
                                                ff_refs,
                                                ff_as,
                                                ff_bs,
                                            )
                                        )
                                        x_latent, x_mean = score_sde.langevin_corrector(
                                            x_latent,
                                            conditioning_ims,
                                            vec_t,
                                            ff_refs,
                                            ff_as,
                                            ff_bs,
                                        )

                                    decoded_256 = vqgan.decode(x_mean)

                                    next_frames = F.interpolate(
                                        decoded_256,
                                        size=(h, w),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                    next_frames = torch.clamp(
                                        next_frames, min=-1.0, max=1.0
                                    )

                                    generated_frames_seq.append(next_frames)
                                    curr_frames = next_frames
                                # =========================================================

                                gen_videos_batch = torch.stack(
                                    generated_frames_seq, dim=2
                                )
                                gen_videos_batch = (
                                    gen_videos_batch.unsqueeze(1).detach().cpu()
                                )

                                for repeat_in_batch in range(current_batch_size):
                                    global_repeat_idx = start_idx + repeat_in_batch
                                    gen_video = gen_videos_batch[repeat_in_batch, 0]
                                    gen_last_frame = gen_video[:, -1, :, :]
                                    gen_last_frames.append(gen_last_frame)

                                    gen_video_post12 = gen_video[:, -12:, :, :]
                                    concat_video = torch.cat(
                                        [gen_video_post12, original_future], dim=1
                                    )
                                    gen_action_single = virtual_actions_batch[
                                        repeat_in_batch
                                    ].to(torch.float32)
                                    gen_action_post12 = gen_action_single[-12:, :]
                                    concat_action = torch.cat(
                                        [gen_action_post12, original_future_actions],
                                        dim=0,
                                    )

                                    gen_videos_all.append(concat_video)
                                    gen_actions_all.append(concat_action.cpu().numpy())

                                    save_video(
                                        rearrange(
                                            init_video_single, "1 c () t h w -> c t h w"
                                        ).cpu(),
                                        os.path.join(
                                            sample_save_dir,
                                            f"input_history_{global_repeat_idx}.mp4",
                                        ),
                                        fps=10,
                                    )
                                    save_video(
                                        gen_video,
                                        os.path.join(
                                            sample_save_dir,
                                            f"virtual_generated_{global_repeat_idx}.mp4",
                                        ),
                                        fps=10,
                                    )
                                    save_video(
                                        concat_video,
                                        os.path.join(
                                            sample_save_dir,
                                            f"augmented_concat_{global_repeat_idx}.mp4",
                                        ),
                                        fps=10,
                                    )
                                    np.save(
                                        os.path.join(
                                            sample_save_dir,
                                            f"virtual_action_{global_repeat_idx}.npy",
                                        ),
                                        concat_action.cpu().numpy(),
                                    )

                                del (
                                    virtual_actions_batch,
                                    init_video_batch,
                                    gen_videos_batch,
                                    poses_4x4,
                                )
                                torch.cuda.empty_cache()
                                gc.collect()

                            real_first_frame = original_future[:, 0, :, :]
                            real_first_frames = [real_first_frame] * total_repeats
                            tsne_results, metrics, top_half_indices = (
                                visualize_tsne_for_frames(
                                    gen_last_frames,
                                    real_first_frames,
                                    total_repeats,
                                    save_path=os.path.join(
                                        sample_save_dir,
                                        "tsne_gen_last_vs_real_first.png",
                                    ),
                                )
                            )

                            for traj_idx in top_half_indices:
                                self.lerobot_episode_counter += 1
                                gen_video_concat = gen_videos_all[traj_idx]
                                gen_action_concat = gen_actions_all[traj_idx]
                                bgr_frames = tensor_video_to_cv2_frames(
                                    gen_video_concat, target_size=(480, 640)
                                )
                                depth_frames = generate_dummy_depth_frames(
                                    len(bgr_frames), size=(480, 640)
                                )

                                gen_action_adapted = []
                                for action_8d in gen_action_concat:
                                    position = action_8d[0:3]
                                    orientation = action_8d[3:7]
                                    gripper = action_8d[7:8]
                                    gen_action_adapted.append(
                                        [position, orientation, gripper]
                                    )

                                self.lerobot_formatter.process_episode(
                                    episode_num=self.lerobot_episode_counter,
                                    bgr_fixed_imgs=bgr_frames,
                                    bgr_hand_imgs=[],
                                    depth_fixed_imgs=[],
                                    depth_hand_imgs=[],
                                    actions=gen_action_adapted,
                                    reward=[],
                                )

                            self.lerobot_formatter.finalize_metadata(
                                task="robot_manipulation"
                            )
                            # =========================================================================

                            del (
                                init_video_single,
                                original_future_video_single,
                                original_future,
                            )
                            del (
                                gen_last_frames,
                                real_first_frames,
                                gen_videos_all,
                                gen_actions_all,
                            )
                            torch.cuda.empty_cache()
                            gc.collect()

                    except Exception as e:
                        accelerator.print(f"处理第 {batch_idx+1} 个batch时出错: {e}")
                        import traceback

                        traceback.print_exc()
                        continue

            self.lerobot_formatter.finalize_metadata(task="robot_manipulation")
            cleanup_distributed()

    def train(self):
        self.augment_dataset(
            save_root="./augment_20_30_20_demo_pick_20260213_1630",
            num_samples=50,
            offset_scale=0.1,
        )


if __name__ == "__main__":
    config_file = "./configs/acwm_cosmos_argument.yaml"
    trainer = GeSimTrainer(config_file=config_file)

    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()

    trainer.train()
