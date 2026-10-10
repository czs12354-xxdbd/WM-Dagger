import os, random, math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib

# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ----------------------------------------------------
from utils.model_utils import (
    load_condition_models,
    load_latent_models,
    load_vae_models,
    load_diffusion_model,
    count_model_parameters,
    unwrap_model,
)
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video

# ----------------------------------------------------
from utils.data_utils import (
    get_latents,
    get_text_conditions,
    gen_noise_from_condition_frame_latent,
    randn_tensor,
    apply_color_jitter_to_video,
)

# ----------------------------------------------------
from utils.extra_utils import act_metric
from utils.get_my_batch_traj import get_batch_arm_traj_maps, simple_radius_gen_func
from utils.get_my_batch_ray import get_ray_maps_batch
from utils.get_ray_maps import get_ray_maps
from utils.geometry_utils import resize_traj_and_ray


import torch.nn.functional as F

LOG_LEVEL = "INFO"
logger = get_logger("gesim_runner")
logger.setLevel(LOG_LEVEL)


def actions_to_homogeneous_matrices(actions_tensor):
    assert (
        len(actions_tensor.shape) == 3 and actions_tensor.shape[-1] == 8
    ), f"输入张量需为 [b, t, 8]，当前形状：{actions_tensor.shape}"

    b, t, _ = actions_tensor.shape
    device = actions_tensor.device
    dtype = actions_tensor.dtype

    translation = actions_tensor[..., :3]
    quaternion_wxyz = actions_tensor[..., 3:7]

    quaternion_wxyz = F.normalize(quaternion_wxyz, p=2, dim=-1)  # [b, t, 4]
    qw, qx, qy, qz = torch.unbind(quaternion_wxyz, dim=-1)

    # https://en.wikipedia.org/wiki/Quaternion#Rotation_matrix
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
    gripper_state = actions_tensor[..., 7]

    gripper_tensor = gripper_state.unsqueeze(1)

    gripper_tensor = gripper_tensor.unsqueeze(2)

    gripper_tensor = gripper_tensor.unsqueeze(-1)

    gripper_tensor = gripper_tensor.unsqueeze(-1)

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

        # self.val_index = []
        # for _ in range(self.args.batch_size):
        #     self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        # if self.state.accelerator.is_main_process:
        #     with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
        #         file.write(", ".join(map(str, self.val_index)))

        # subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        # self.val_dataloader = torch.utils.data.DataLoader(
        #     subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", True)
        # )
        # logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=getattr(self.args, "val_shuffle", True),
        )
        logger.info(
            f">>>>>>>>>>>>>Total Validation Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n"
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
        ###
        # self.vae.temperal_downsample = [False, True, True]
        ###
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
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # logger.info("Initializing trainable parameters")

        # for component in components_to_disable_grads:
        #     if component is not None:
        #         component.requires_grad_(False)

        # if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
        #     raise ValueError(
        #         "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        #     )

        # if self.args.gradient_checkpointing:

        # if self.args.allow_tf32 and torch.cuda.is_available():
        #     torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
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

        # logger.info("Initializing optimizer and lr scheduler")

        # train_mode = self.args.train_mode

        # self.state.train_epochs = self.args.train_epochs
        # self.state.train_steps = self.args.train_steps

        # if self.args.mixed_precision == "fp16":
        #     cast_training_params([self.transformer], dtype=torch.float32)

        # self.state.learning_rate = self.args.lr
        # if self.args.scale_lr:
        #     self.state.learning_rate = (
        #         self.state.learning_rate
        #         * self.args.gradient_accumulation_steps
        #         * self.args.batch_size
        #         * self.state.accelerator.num_processes
        #     )

        # transformer_trainable_params = []
        # if train_mode == 'action_only':
        #     for name, param in self.transformer.named_parameters():
        #         if 'action_' in name:
        #             param.requires_grad = True
        #             transformer_trainable_params.append(param)
        #         else:
        #             param.requires_grad = False
        # elif train_mode == "video_only":
        #     for name, param in self.transformer.named_parameters():
        #         if 'action_' not in name:
        #             param.requires_grad = True
        #             transformer_trainable_params.append(param)
        #         else:
        #             param.requires_grad = False
        # elif train_mode == "all" or train_mode == 'action_full':
        #     for name, param in self.transformer.named_parameters():
        #         param.requires_grad = True
        #         transformer_trainable_params.append(param)
        # else:
        #     raise NotImplementedError

        # num_trainable_params = sum(p.numel() for p in transformer_trainable_params)
        # logger.info(f'Total trainable parameters: {num_trainable_params}')

        # transformer_parameters_with_lr = {
        #     "params": transformer_trainable_params,
        #     "lr": self.state.learning_rate,
        # }
        # params_to_optimize = [transformer_parameters_with_lr]
        # self.state.num_trainable_parameters = sum(p.numel() for p in transformer_trainable_params)

        # optimizer = get_optimizer(
        #     params_to_optimize=params_to_optimize,
        #     optimizer_name=self.args.optimizer,
        #     learning_rate=self.args.lr,
        #     beta1=self.args.beta1,
        #     beta2=self.args.beta2,
        #     beta3=self.args.beta3,
        #     epsilon=self.args.epsilon,
        #     weight_decay=self.args.weight_decay,
        #     use_8bit = self.args.optimizer_8bit,
        #     use_torchao = self.args.optimizer_torchao,
        # )

        # num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        # if self.state.train_steps is None:
        #     self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
        #     self.state.overwrote_max_train_steps = True

        # lr_scheduler = get_scheduler(
        #     name=self.args.lr_scheduler,
        #     optimizer=optimizer,
        #     num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
        #     num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
        #     num_cycles=self.args.lr_num_cycles,
        #     power=self.args.lr_power,
        # )

        # self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler

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

    def train(self):
        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        # if self.scheduler is not None:
        self.scheduler.register_to_config(
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            sigma_data=self.sigma_data,
            final_sigmas_type=self.final_sigmas_type,
        )

        logger.info("Starting GE-sim training")
        memory_statistics = get_memory_statistics()
        logger.info(
            f"Memory before training start: {json.dumps(memory_statistics, indent=4)}"
        )

        self.state.train_batch_size = (
            self.args.batch_size
            * self.state.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(
            device=accelerator.device, dtype=weight_dtype
        )
        # scheduler_sigmas =0.002+scheduler_sigmas*(80-0.002)
        # print('scheduler_sigmas:',scheduler_sigmas)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        anomalies = []

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.diffusion_model.train()

            running_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate([self.diffusion_model]):

                    video = batch["video"]  # shape: b, c, v, t, h, w
                    actions = batch["actions"]  # shape: b, t_act, c_act
                    # extrinsics = batch['extrinsics']  # shape: b, v, t, 4, 4
                    # intrinsics = batch['intrinsics']  # shape: b, v, 3, 3
                    captions = batch["caption"]

                    video = video.to(
                        accelerator.device, dtype=weight_dtype
                    ).contiguous()
                    batch_size, c, n_view, total_frames, h, w = video.shape
                    video = rearrange(
                        video, "b c v t h w -> (b v) c t h w"
                    )  # (b*v) c t h w

                    if self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video)

                    mem_size = self.args.data["train"]["n_previous"]
                    n_prev = mem_size
                    mem = video[:, :, :mem_size]
                    future_video = video[:, :, mem_size:]
                    _, c, t_future, h, w = future_video.shape

                    # print('mem_size:',mem_size,'video shape:',video.shape,'future_video shape:',future_video.shape)
                    actions_tensor = actions.to(
                        accelerator.device, dtype=weight_dtype
                    )  # b t 8
                    extrinsics_tensor = actions_to_homogeneous_matrices(actions_tensor)
                    # print('extrinsics_tensor:',extrinsics_tensor)
                    c2w = np.array(
                        [
                            [0.0000000, 0.5735659, -0.8191594, 3.5],
                            [1.0000000, 0.0000000, 0.0000000, 0.5],
                            [0.0000000, -0.8191594, -0.5735659, 2.09],
                            [0.0000000, 0.0000000, 0.0000000, 1],
                        ]
                    )
                    intrinsic = np.array(
                        [
                            [796.8544625226342, 0.0, 1354.4265245977356],
                            [0.0, 794.4770470238264, 1011.4847310011687],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    # extrinsics_tensor = self.expand_to_bvt(c2w, batch_size, n_view, total_frames)
                    intrinsics_tensor = self.expand_to_bvt(
                        intrinsic, batch_size, n_view, total_frames
                    )
                    intrinsics_tensor = torch.tensor(intrinsics_tensor).to(
                        accelerator.device, dtype=weight_dtype
                    )
                    # print('intrinsics_tensor:',intrinsics_tensor)
                    trajs = gripper_state_to_tensor(actions_tensor).repeat(
                        1, 3, 1, 1, 1, 1
                    )
                    trajs = trajs * 2 - 1
                    print("trajs.shape:", trajs.shape)
                    rays_o, rays_d = get_ray_maps(
                        intrinsics_tensor, extrinsics_tensor, H=h, W=w
                    )
                    # print("rays_o.shape",rays_o.shape,rays_d.shape)
                    rays = torch.cat((rays_o, rays_d), dim=-1).reshape(
                        batch_size, n_view, total_frames, h, w, -1
                    )
                    rays = rays.permute(0, 5, 1, 2, 3, 4).to(
                        accelerator.device
                    )  # (b, c, v, t, h, w)
                    print("rays.shape:", rays.shape)
                    cond_to_concat = torch.cat(
                        (trajs, rays), dim=1
                    )  # (b,c_total, v, t, h, w)
                    cond_to_concat = rearrange(
                        cond_to_concat, "b c v t h w -> (b v) c t h w", b=batch_size
                    )

                    _, _, raw_frames, raw_height, raw_width = future_video.shape
                    latent_frames = (
                        raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
                    )
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    n_fut = (raw_frames - 1) // self.TEMPORAL_DOWN_RATIO + 1
                    cond_to_concat_resized = resize_traj_and_ray(
                        cond_to_concat,
                        mem_size=mem_size,
                        future_size=n_fut,
                        height=latent_height,
                        width=latent_width,
                    )

                    dropout_factor = torch.rand(batch_size).to(
                        accelerator.device, dtype=weight_dtype
                    )
                    dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                    dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, future_video
                    )

                    mem_latents = rearrange(
                        mem_latents,
                        "(b v m) (h w) c -> (b v) c m h w",
                        b=batch_size,
                        m=mem_size,
                        h=latent_height,
                    )
                    future_video_latents = rearrange(
                        future_video_latents,
                        "(b v) (f h w) c -> (b v) c f h w",
                        b=batch_size,
                        h=latent_height,
                        w=latent_width,
                    )

                    text_conds = get_text_conditions(
                        self.tokenizer, self.text_encoder, captions
                    )
                    prompt_embeds = text_conds["prompt_embeds"]
                    prompt_attention_mask = text_conds["prompt_attention_mask"]
                    prompt_embeds = (
                        self.uncond_prompt_embeds.repeat(batch_size, 1, 1)
                        * dropout_mask_prompt
                        + prompt_embeds * ~dropout_mask_prompt
                    )

                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    action_indices = (
                        action_weights * self.scheduler.config.num_train_timesteps
                    ).long()
                    action_sigmas = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long()

                    weights = action_weights.unsqueeze(1).repeat(1, n_view)
                    weights = rearrange(weights, "b v -> (b v)")
                    indices = (
                        weights * self.scheduler.config.num_train_timesteps
                    ).long()
                    sigmas = scheduler_sigmas[indices]
                    print("sigmas:", sigmas, "sigmas")
                    timesteps = sigmas  # / (sigmas + 1)
                    print("timesteps11", timesteps, timesteps.shape, "timesteps11")
                    if self.args.return_action:
                        if getattr(self.args, "add_state", False):
                            act_state = (
                                batch["state"]
                                .to(accelerator.device, dtype=weight_dtype)
                                .contiguous()
                            )
                        else:
                            act_state = None

                        actions = (
                            actions[:, -self.args.data["train"]["action_chunk"] :]
                            .to(accelerator.device, dtype=weight_dtype)
                            .contiguous()
                        )
                        action_dim = actions.shape[-1]
                        noise_actions = randn_tensor(
                            actions.shape, device=accelerator.device, dtype=weight_dtype
                        )

                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(
                            1, actions.shape[1]
                        )
                        action_ss = action_sigmas.reshape(-1, 1, 1).repeat(
                            1, 1, actions.shape[-1]
                        )
                        noisy_actions = (
                            1.0 - action_ss
                        ) * actions + action_ss * noise_actions

                        action_weights = (
                            compute_loss_weighting_for_sd3(
                                weighting_scheme=self.args.flow_weighting_scheme,
                                sigmas=action_sigmas,
                            )
                            .reshape(-1, 1, 1)
                            .repeat(1, 1, actions.size(-1))
                        )
                    else:
                        actions = None
                        action_timesteps = None
                        noisy_actions = None
                        act_state = None

                    noise, conditioning_mask, cond_indicator = (
                        gen_noise_from_condition_frame_latent(
                            future_video_latents,
                            latent_frames,
                            latent_height,
                            latent_width,
                            # mem_con_latents, latent_frames, latent_height, latent_width,
                            noise_to_condition_frames=self.args.noise_to_first_frame,
                        )
                    )

                    device = accelerator.device
                    weight_dtype = weight_dtype
                    future_latents = future_video_latents
                    batch_size_latent = future_latents.shape[0]
                    # timestep_weights = compute_density_for_timestep_sampling(
                    #     batch_size=batch_size_latent,
                    #     logit_mean=1.0,
                    #     logit_std=1.0,
                    # ).to(device)
                    # sigmas = sigma_min * (sigma_max / sigma_min) ** timestep_weights
                    t = timesteps
                    t_expanded = timesteps[:, None, None, None, None]

                    noise = torch.randn_like(future_latents, dtype=weight_dtype)
                    # noisy_future_latents = (1 - t.view(-1, 1, 1, 1, 1)) * future_latents + t.view(-1, 1, 1, 1, 1) * noise

                    ss = timesteps.reshape(
                        -1, 1, 1, 1, 1
                    )  # .repeat(1, future_latents.size(1),1, 1, 1, 1)
                    if self.args.return_action and self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_future_latents = (1.0 - ss) * future_latents + ss * noise

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss, shape bv,1,c
                    weights = (
                        compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme,
                            sigmas=sigmas,
                        )
                        .reshape(-1, 1, 1)
                        .repeat(1, 1, future_latents.size(-1))
                    )

                    cond_latent = torch.cat(
                        [mem_latents, noisy_future_latents], dim=2
                    )  # (b v) c (n_prev + t_fut_latent) h w

                    cond_latent = torch.cat(
                        [cond_latent, cond_to_concat_resized], dim=1
                    )  # (b v) (c + c_cond) T h w

                    batch_size_latent = future_latents.shape[0]
                    cond_indicator = torch.zeros(
                        batch_size_latent,
                        1,
                        cond_latent.shape[2],
                        1,
                        1,
                        device=device,
                        dtype=weight_dtype,
                    )
                    cond_indicator[:, :, :n_prev] = 1.0
                    sigma_conditioning = torch.tensor(
                        0.0001, device=device, dtype=weight_dtype
                    )
                    t_conditioning = sigma_conditioning / (sigma_conditioning + 1)
                    cond_timestep = (
                        cond_indicator * t_conditioning
                        + (1 - cond_indicator) * t_expanded
                    )

                    ones_padding = torch.ones_like(
                        cond_latent[:, :1], dtype=weight_dtype
                    )
                    zeros_padding = torch.zeros_like(
                        cond_latent[:, :1], dtype=weight_dtype
                    )
                    condition_mask = (
                        cond_indicator * ones_padding
                        + (1 - cond_indicator) * zeros_padding
                    )
                    padding_mask = torch.zeros(
                        1,
                        1,
                        cond_latent.shape[-2],
                        cond_latent.shape[-1],
                        device=device,
                        dtype=weight_dtype,
                    )

                    print("timesteps", timesteps, timesteps.shape, "timesteps")
                    model_output = self.diffusion_model(
                        hidden_states=cond_latent.to(dtype=self.state.weight_dtype),
                        timestep=cond_timestep,
                        encoder_hidden_states=prompt_embeds.to(
                            dtype=self.state.weight_dtype
                        ),
                        fps=16,
                        condition_mask=condition_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        height=cond_latent.shape[-2],
                        width=cond_latent.shape[-1],
                        n_view=n_view,
                        num_frames=n_fut + n_prev,
                        return_video=True,
                    )[0]["video"]

                    model_pred = model_output[:, :, n_prev:]
                    # print('model_pred:',model_pred,model_pred.shape,000000000000000)
                    # pred_all = forward_pass(
                    #     timesteps=timesteps,
                    #     noisy_latents=latents.to(dtype=self.state.weight_dtype),
                    #     prompt_embeds=prompt_embeds.to(dtype=self.state.weight_dtype),
                    #     prompt_attention_mask=prompt_attention_mask.to(dtype=self.state.weight_dtype),
                    #     num_frames=latent_frames,
                    #     height=latent_height,
                    #     width=latent_width,
                    #     n_view=n_view,
                    #     action_states=noisy_actions,
                    #     action_timestep=action_timesteps,
                    #     return_video=self.args.return_video or self.args.return_action,
                    #     return_action=self.args.return_action,
                    #     video_attention_mask=None,
                    #     history_action_state=act_state,
                    #     condition_mask=conditioning_mask.to(dtype=self.state.weight_dtype),
                    # )['latents']

                    if (
                        self.args.train_mode == "all"
                        or self.args.train_mode == "video_only"
                    ):
                        c_skip = 1 - t.view(-1, 1, 1, 1, 1)
                        c_out = -t.view(-1, 1, 1, 1, 1)
                        # c_skip = 1 - t
                        # c_out = -t
                        target = noise - future_latents
                        # if loss_type == "mse":
                        #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        # elif loss_type == "l1":
                        #     loss = F.l1_loss(model_pred.float(), target.float(), reduction="mean")
                        loss_video = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                        # pred = pred_all['video']
                        # target = noise - future_video_latents

                        # print('weights:',weights.shape)
                        # print('pred:',pred.shape)
                        # print('target:',target.shape)
                        # print('model_pred shape:',model_pred.shape,8888888888)
                        # loss_video = (model_pred.float() - target.float()).pow(2) #weights.float() * (model_pred.float() - target.float()).pow(2)
                        # loss_video = loss_video * (1 - condition_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1)))
                        # loss_video = loss_video.mean(list(range(1, loss_video.ndim)))
                        # loss_video = loss_video.mean()
                    else:
                        loss_video = 0.0

                    if (
                        self.args.train_mode == "all"
                        or self.args.train_mode == "action_only"
                        or self.args.train_mode == "action_full"
                    ):
                        target_action = noise_actions - actions
                        loss_action = action_weights.float() * (
                            pred_all["action"].float() - target_action.float()
                        ).pow(2)
                        loss_action = loss_action.mean()
                    else:
                        loss_action = 0.0

                    action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)
                    loss = loss_video + action_loss_scale * loss_action

                    assert torch.isnan(loss) == False, "NaN loss detected"
                    accelerator.backward(loss)
                    if (
                        accelerator.sync_gradients
                        and accelerator.distributed_type != DistributedType.DEEPSPEED
                    ):
                        grad_norm = accelerator.clip_grad_norm_(
                            self.diffusion_model.parameters(), self.args.max_grad_norm
                        )
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                #     loss = loss_video + action_loss_scale * loss_action

                #     assert torch.isnan(loss) == False, "NaN loss detected"
                #     accelerator.backward(loss)
                #     if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                #         grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                #         logs["grad_norm"] = grad_norm
                #     self.optimizer.step()
                #     self.lr_scheduler.step()
                #     self.optimizer.zero_grad()

                loss = accelerator.reduce(loss.detach(), reduction="mean")
                if self.args.train_mode in ["all", "action_only", "action_full"]:
                    loss_action = accelerator.reduce(
                        loss_action.detach(), reduction="mean"
                    )
                if self.args.train_mode in ["all", "video_only"]:
                    loss_video = accelerator.reduce(
                        loss_video.detach(), reduction="mean"
                    )

                running_loss += loss.item()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process and self.writer is not None:
                        self.writer.add_scalar(
                            "Training Loss", loss.item(), global_step
                        )
                        if self.args.train_mode in [
                            "all",
                            "action_only",
                            "action_full",
                        ]:
                            self.writer.add_scalar(
                                "Action loss", loss_action.mean().item(), global_step
                            )
                        if self.args.train_mode in ["all", "video_only"]:
                            self.writer.add_scalar(
                                "Video loss", loss_video.item(), global_step
                            )

                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(
                            self.save_folder, f"Validation_step_{global_step}"
                        )
                        self.validate(
                            accelerator,
                            model_save_dir,
                            global_step,
                            n_view=n_view,
                            n_chunk=1,
                        )

                if global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        dtype = (
                            torch.float16
                            if self.args.mixed_precision == "fp16"
                            else (
                                torch.bfloat16
                                if self.args.mixed_precision == "bf16"
                                else torch.float32
                            )
                        )

                        model_save_dir = os.path.join(
                            self.save_folder, f"step_{global_step}"
                        )
                        model_to_save.save_pretrained(
                            model_save_dir, safe_serialization=True
                        )
                        del model_to_save

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

            if global_step >= self.state.train_steps:
                logger.info(">>> max train step reached")
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else (
                    torch.bfloat16
                    if self.args.mixed_precision == "bf16"
                    else torch.float32
                )
            )

            model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
            self.diffusion_model.save_pretrained(
                model_save_dir, safe_serialization=True
            )

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(
            f"Memory after training end: {json.dumps(memory_statistics, indent=4)}"
        )

        accelerator.end_training()

    # def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30, image=None, prompt=None, cap=None, path=None, gt_actions=None, to_log=True):
    #     """
    #     """
    #     os.makedirs(model_save_dir, exist_ok=True)

    #     pipe = self.pipeline_class(
    #         scheduler = self.scheduler, vae = self.vae, text_encoder = self.text_encoder, tokenizer = self.tokenizer,
    #         transformer = unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
    #     )

    #     if image is None:
    #         batch = next(iter(self.val_dataloader))
    #         prompt = batch['caption']

    #             loss = accelerator.reduce(loss.detach(), reduction='mean')
    #             if self.args.train_mode in ['all', 'action_only', 'action_full']:
    #                 loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
    #             if self.args.train_mode in ['all', 'video_only']:
    #                 loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')

    #             running_loss += loss.item()

    #             if accelerator.sync_gradients:
    #                 progress_bar.update(1)
    #                 global_step += 1

    #             logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
    #             progress_bar.set_postfix(logs)
    #             accelerator.log(logs, step=global_step)

    #             if global_step % self.args.steps_to_log == 0:
    #                 if accelerator.is_main_process and self.writer is not None:
    #                     self.writer.add_scalar("Training Loss", loss.item(), global_step)
    #                     if self.args.train_mode in ['all', 'action_only', 'action_full']:
    #                         self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
    #                     if self.args.train_mode in ['all', 'video_only']:
    #                         self.writer.add_scalar("Video loss", loss_video.item(), global_step)

    #             if global_step % self.args.steps_to_val == 0:
    #                 accelerator.wait_for_everyone()
    #                 if accelerator.is_main_process:
    #                     model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
    #                     self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)

    #             if global_step % self.args.steps_to_save == 0:
    #                 accelerator.wait_for_everyone()
    #                 if accelerator.is_main_process:
    #                     model_to_save = unwrap_model(accelerator, self.diffusion_model)
    #                     dtype = (
    #                         torch.float16
    #                         if self.args.mixed_precision == "fp16"
    #                         else torch.bfloat16
    #                         if self.args.mixed_precision == "bf16"
    #                         else torch.float32
    #                     )

    #                     model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
    #                     model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
    #                     del model_to_save

    #         memory_statistics = get_memory_statistics()
    #         logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

    #         if accelerator.is_main_process and self.writer is not None:
    #             avg_loss = running_loss / len(self.train_dataloader)
    #             self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

    #         if global_step >= self.state.train_steps:
    #             logger.info(">>> max train step reached")
    #             break

    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
    #         dtype = (
    #             torch.float16
    #             if self.args.mixed_precision == "fp16"
    #             else torch.bfloat16
    #             if self.args.mixed_precision == "bf16"
    #             else torch.float32
    #         )

    #         model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
    #         self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

    #     del self.diffusion_model, self.scheduler
    #     free_memory()
    #     memory_statistics = get_memory_statistics()
    #     logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

    #     accelerator.end_training()

    def validate(
        self,
        accelerator,
        model_save_dir,
        global_step,
        n_view=1,
        n_chunk=30,
        image=None,
        prompt=None,
        cap=None,
        path=None,
        gt_actions=None,
        to_log=True,
    ):
        os.makedirs(model_save_dir, exist_ok=True)

        pipe = self.pipeline_class(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=(
                unwrap_model(accelerator, self.diffusion_model)
                if accelerator is not None
                else self.diffusion_model
            ),
        )

        if image is None:
            batch = next(iter(self.val_dataloader))
            image = batch["video"][:, :, :, : self.args.data["val"]["n_previous"]]
            prompt = batch["caption"]
            gt_video = batch["video"]
            actions = batch["actions"]
            gt_actions = batch["actions"][:, -self.args.data["val"]["action_chunk"] :]

        b, c, v, t_mem, h, w = image.shape
        mem_size = self.args.data["val"]["n_previous"]
        future_size = self.args.data["val"]["chunk"]
        print("future_size", future_size)
        batch_size = min(b, self.args.batch_size)

        image = image[:batch_size]
        actions = actions[:batch_size] if actions is not None else None
        gt_video = gt_video[:batch_size] if gt_video is not None else None
        gt_actions = gt_actions[:batch_size] if gt_actions is not None else None
        # extrinsics = extrinsics[:batch_size] if extrinsics is not None else None
        # intrinsics = intrinsics[:batch_size] if intrinsics is not None else None
        prompt = prompt[:batch_size] if prompt is not None else None

        actions_tensor = actions.to(accelerator.device)  # b t 8
        extrinsics_tensor = actions_to_homogeneous_matrices(actions_tensor).to(
            accelerator.device, dtype=self.state.weight_dtype
        )
        intrinsic = np.array(
            [
                [796.8544625226342, 0.0, 1354.4265245977356],
                [0.0, 794.4770470238264, 1011.4847310011687],
                [0.0, 0.0, 1.0],
            ]
        )
        # extrinsics_tensor = self.expand_to_bvt(c2w, batch_size, n_view, total_frames)
        intrinsics_tensor = self.expand_to_bvt(
            intrinsic, batch_size, n_view, future_size + mem_size
        )
        intrinsics_tensor = torch.tensor(intrinsics_tensor).to(
            accelerator.device, dtype=self.state.weight_dtype
        )
        # print('intrinsics_tensor:',intrinsics_tensor)
        trajs = gripper_state_to_tensor(actions_tensor).repeat(1, 3, 1, 1, 1, 1)
        trajs = trajs * 2 - 1
        print("trajs.shape:", trajs.shape)
        rays_o, rays_d = get_ray_maps(intrinsics_tensor, extrinsics_tensor, H=h, W=w)
        rays = torch.cat((rays_o, rays_d), dim=-1).reshape(
            batch_size, v, future_size + mem_size, h, w, -1
        )
        rays = rays.permute(0, 5, 1, 2, 3, 4).to(
            accelerator.device
        )  # (b, c, v, t, h, w)
        print("rays.shape", rays.shape)

        cond_to_concat = torch.cat((trajs, rays), dim=1)  # (b, c_total, v, t, h, w)
        cond_to_concat = rearrange(cond_to_concat, "b c v t h w -> (b v) c t h w")

        n_fut = (future_size - 1) // self.TEMPORAL_DOWN_RATIO + 1
        cond_to_concat_resized = resize_traj_and_ray(
            cond_to_concat,
            mem_size=mem_size,
            future_size=n_fut,
            height=h // self.SPATIAL_DOWN_RATIO,
            width=w // self.SPATIAL_DOWN_RATIO,
        )

        history_action_state = None
        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = (
                batch["state"][:batch_size] if "state" in batch else None
            )

        image_input = rearrange(image, "b c v t h w -> (b v) t c h w")
        print("image_input:", image_input, 1111111111111)

        preds = pipe.infer(
            video=image_input,  # -> v, t, c, h, w
            cond_to_concat=cond_to_concat_resized,
            prompt=prompt,
            negative_prompt="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
            height=h,
            width=w,
            n_view=v,
            num_frames=self.args.data["train"]["chunk"],
            num_inference_steps=self.args.num_inference_step,
            # decode_timestep=0.03,
            # decode_noise_scale=0.025,
            n_prev=self.args.data["train"]["n_previous"],
            guidance_scale=1.0,
            merge_view_into_width=False,
            output_type="pt",
            postprocess_video=False,
        )  # preds: v c t h w , range -1 to 1 (could exceed range)

        # preds = pipe.infer(
        #     video=image_input,
        #     prompt=prompt,
        #     negative_prompt='',
        #     num_inference_steps=self.args.num_inference_step,
        #     # decode_timestep=0.03,
        #     # decode_noise_scale=0.025,
        #     guidance_scale=1.0,
        #     output_type="pt",
        #     height=h,
        #     width=w,
        #     n_view=v,
        #     return_action=self.args.return_action,
        #     n_prev=mem_size,
        #     chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
        #     return_video=self.args.return_video,
        #     noise_seed=42,
        #     action_chunk=self.args.data['train']['action_chunk'],
        #     history_action_state=history_action_state,
        #     pixel_wise_timestep=self.args.pixel_wise_timestep,
        #     n_chunk=n_chunk,
        #     action_dim=self.args.diffusion_model["config"]["action_in_channels"],
        # )

        if cap is None:
            cap = f"Val_step_{global_step}"

        if gt_video is not None and self.args.return_video:
            gt_video_vis = rearrange(
                gt_video[0].data.cpu(), "c v t h w -> c t h (v w)", v=v
            )
            save_video(
                gt_video_vis[:, mem_size::, ...],
                os.path.join(model_save_dir, f"{cap}_gt.mp4"),
                fps=10,
            )
        print(
            "preds[frames] shape:",
            preds["frames"][0].data.cpu(),
            "preds[frames][0]:",
            55555555555555555,
        )
        if self.args.return_video and "frames" in preds:
            # pred_video = rearrange(preds['frames'][0].data.cpu(), '(b v) c t h w -> b c t h (v w)', b=batch_size, v=v)
            pred_video = preds["frames"][0].data.cpu()
            pred_video = torch.clamp(pred_video, min=-1, max=1)
            # pred_video = rearrange(pred_video, 't c h w -> c t h w')
            print("pred_video shape:", pred_video.shape, 9999999999)
            save_video(
                pred_video, os.path.join(model_save_dir, f"{cap}_pred.mp4"), fps=10
            )
        virtual_action_np = actions_tensor.to(torch.float32).cpu().numpy()
        np.save(os.path.join(model_save_dir, f"virtual_action.npy"), virtual_action_np)
        if to_log and self.writer is not None and prompt:
            self.writer.add_text(f"{cap}/prompt", prompt[0], global_step)

        if self.args.return_action and "action" in preds and gt_actions is not None:
            action_dim = gt_actions.shape[-1]
            action_metrics = act_metric(
                preds["action"][:, :, :action_dim].detach().cpu().float().numpy(),
                gt_actions.detach().cpu().float().numpy(),
                prefix=cap,
                start_stop_interval=[
                    (0, 1),
                    (1, 9),
                    (9, 25),
                    (25, self.args.data["train"]["action_chunk"]),
                ],
            )

            if to_log and self.writer is not None:
                for key, value in action_metrics.items():
                    self.writer.add_scalar(key, value, global_step)

        del cond_to_concat, cond_to_concat_resized, trajs, rays
        torch.cuda.empty_cache()
