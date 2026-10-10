import json
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime


class LerobotDataFormatter:
    def __init__(
        self, output_dir, task_name="robot_manipulation", chunk_size=5000, action_dim=8
    ):
        self.action_dim = action_dim
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.chunk_size = chunk_size

        self.meta_dir = self.output_dir / "meta"
        self.videos_dir = self.output_dir / "videos"
        self.data_dir = self.output_dir / "data"

        for dir_path in [self.meta_dir, self.videos_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.episodes_metadata = []
        self.all_episodes_data = []
        self.episode_index = 0
        self.total_episodes = 0
        self.total_frames = 0

        self._init_modality_config()

    def _init_modality_config(self):
        action_dim = self.action_dim
        modalities = {
            "video": {
                "bgr_fixed": {
                    "original_key": "images.bgr_fixed",
                },
                "bgr_hand": {
                    "original_key": "images.bgr_hand",
                },
                "depth_fixed": {
                    "original_key": "images.depth_fixed",
                },
                "depth_hand": {
                    "original_key": "images.depth_hand",
                },
            },
            "state": {
                "pose": {"original_key": "state.pose", "start": 0, "end": action_dim}
            },
            "action": {"pose": {"start": 0, "end": action_dim}},
            "annotation": {
                "task_description": {},
            },
        }

        with open(self.meta_dir / "modality.json", "w") as f:
            json.dump(modalities, f, indent=2)

    def normalize_image_dimensions(self, images, is_depth=False):
        normalized = []
        for img in images:
            img_np = np.asarray(img)
            if is_depth:
                if len(img_np.shape) == 3:
                    if img_np.shape[2] == 1:
                        img_np = img_np.squeeze(axis=2)
                    else:
                        raise ValueError(
                            f"深度图通道数错误，预期1，实际{img_np.shape[2]}"
                        )
                elif len(img_np.shape) != 2:
                    raise ValueError(
                        f"深度图维度错误，预期2或3，实际{len(img_np.shape)}"
                    )
            else:
                if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                    raise ValueError(
                        f"BGR图维度错误，预期(480,640,3)，实际{img_np.shape}"
                    )
            normalized.append(img_np)
        return normalized

    def save_images_to_video(self, images, output_path, fps=30):
        if not images:
            return

        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for img in images:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            out.write(img)

        out.release()

    def save_depth_images_to_video(self, depth_images, output_path, fps=30):
        if not depth_images:
            return

        height, width = depth_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_depth = np.concatenate([img.flatten() for img in depth_images])
        min_depth = np.percentile(all_depth, 1)
        max_depth = np.percentile(all_depth, 99)

        for depth_img in depth_images:
            normalized = (
                (depth_img - min_depth) / (max_depth - min_depth + 1e-8) * 255
            ).clip(0, 255)
            normalized = normalized.astype(np.uint8)
            bgr_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            out.write(bgr_img)

        out.release()

    def save_chunk(self, episode_data, chunk_num, episode_index):
        chunk_dir = self.data_dir / f"chunk-{chunk_num:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(episode_data)

        df["action"] = df["action"].apply(lambda x: np.array(x, dtype=np.float32))
        df["state.pose"] = df["state.pose"].apply(
            lambda x: np.array(x, dtype=np.float32)
        )

        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"✅ 已保存数值格式数据块 {chunk_num} 到 {parquet_path}")

        # parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        # df.to_parquet(parquet_path, index=False)

    def finalize_metadata(self, task):
        action_dim = self.action_dim
        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for meta in self.episodes_metadata:
                f.write(json.dumps(meta) + "\n")

        chunk_num = (self.total_episodes) // self.chunk_size
        info = {
            "codebase_version": "v2.0",
            "robot_type": "irb6700",
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": 4,
            "total_chunks": chunk_num,
            "chunks_size": self.chunk_size,
            "fps": 30.0,
            "splits": {"train": f"0:{self.total_episodes-1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "images.bgr_fixed": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 30.0,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": "false",
                        "has_audio": "false",
                    },
                },
                "images.bgr_hand": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 30.0,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": "false",
                        "has_audio": "false",
                    },
                },
                "images.depth_fixed": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 30.0,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": "true",
                        "has_audio": "false",
                    },
                },
                "images.depth_hand": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 30.0,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": "true",
                        "has_audio": "false",
                    },
                },
                "state.pose": {"dtype": "float64", "shape": [action_dim], "names": []},
                "action.pose": {"dtype": "float64", "shape": [action_dim], "names": []},
                "timestamp": {"dtype": "float64", "shape": [1]},
                "annotation.task_description": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "annotation.human.validity": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.reward": {"dtype": "float64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
            },
        }

        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        tasks_ = {"task_index": 0, "task": task}
        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            json.dump(tasks_, f)

        print(f"转换完成! 共处理 {self.episode_index} 个episode")
        print(f"结果保存在: {self.output_dir}")

    def process_episode(
        self,
        episode_num,
        bgr_fixed_imgs,
        bgr_hand_imgs,
        depth_fixed_imgs,
        depth_hand_imgs,
        actions,
        reward=[],
    ):
        # try:
        self.total_episodes += 1

        lengths = [
            len(bgr_fixed_imgs),
            len(bgr_hand_imgs),
            len(depth_fixed_imgs),
            len(depth_hand_imgs),
            len(actions),
        ]

        # if len(set(lengths)) != 1:
        #     return

        num_frames = lengths[0]
        if num_frames == 0:
            print(f"警告: Episode {episode_num} 中没有数据，已跳过")
            return

        self.total_frames += num_frames

        bgr_fixed_imgs = self.normalize_image_dimensions(bgr_fixed_imgs, is_depth=False)
        bgr_hand_imgs = self.normalize_image_dimensions(bgr_hand_imgs, is_depth=False)
        depth_fixed_imgs = self.normalize_image_dimensions(
            depth_fixed_imgs, is_depth=True
        )
        depth_hand_imgs = self.normalize_image_dimensions(
            depth_hand_imgs, is_depth=True
        )

        chunk_num = (self.episode_index) // self.chunk_size

        for cam_type in ["bgr_fixed", "bgr_hand", "depth_fixed", "depth_hand"]:
            (self.videos_dir / f"chunk-{chunk_num:03d}/images.{cam_type}").mkdir(
                parents=True, exist_ok=True
            )

        video_paths = {}

        video_path = (
            self.videos_dir
            / f"chunk-{chunk_num:03d}/images.bgr_fixed/episode_{self.episode_index:06d}.mp4"
        )
        self.save_images_to_video(bgr_fixed_imgs, video_path)
        video_paths["bgr_fixed"] = str(video_path)

        video_path = (
            self.videos_dir
            / f"chunk-{chunk_num:03d}/images.bgr_hand/episode_{self.episode_index:06d}.mp4"
        )
        self.save_images_to_video(bgr_hand_imgs, video_path)
        video_paths["bgr_hand"] = str(video_path)

        video_path = (
            self.videos_dir
            / f"chunk-{chunk_num:03d}/images.depth_fixed/episode_{self.episode_index:06d}.mp4"
        )
        self.save_depth_images_to_video(depth_fixed_imgs, video_path)
        video_paths["depth_fixed"] = str(video_path)

        video_path = (
            self.videos_dir
            / f"chunk-{chunk_num:03d}/images.depth_hand/episode_{self.episode_index:06d}.mp4"
        )
        self.save_depth_images_to_video(depth_hand_imgs, video_path)
        video_paths["depth_hand"] = str(video_path)

        episode_metadata = {
            "episode_index": self.episode_index,
            "task_index": 0,
            "length": num_frames,
            "video_paths": {
                "video.images.bgr_fixed": video_paths["bgr_fixed"],
                "video.images.bgr_hand": video_paths["bgr_hand"],
                "video.images.depth_fixed": video_paths["depth_fixed"],
                "video.images.depth_hand": video_paths["depth_hand"],
            },
            "timestamp": datetime.now().isoformat(),
            "source": f"Episode_{episode_num}",
        }
        self.episodes_metadata.append(episode_metadata)

        episode_data = []
        for step in range(num_frames):
            action_pair = actions[step]
            # print('action_pair:',action_pair)
            # if len(action_pair) != 3:

            position = np.asarray(action_pair[0]).flatten()
            orientation = np.asarray(action_pair[1]).flatten()
            griper_state = np.asarray(action_pair[2]).flatten()

            # if position.shape != (3,):
            # if orientation.shape != (4,):

            action = np.concatenate([position, orientation, griper_state])
            # print('action:',action)
            if action.shape[0] != 8:
                raise ValueError(f"维度错误！预期 8，实际得到 {action.shape[0]}")
            # print(action.tolist())
            if reward != []:
                reward_step = reward[step]
            else:
                reward_step = 0
            step_data = {
                "episode_index": self.episode_index,
                "index": step,
                "action": action.tolist(),
                "state.pose": action.tolist(),
                "annotation.task_description": 0,
                "timestamp": step / 30.0,
                "next.done": (step == num_frames - 1),
                "next.reward": reward_step,
            }
            # print(step_data)
            episode_data.append(step_data)

        self.all_episodes_data.extend(episode_data)

        self.save_chunk(episode_data, chunk_num, self.episode_index)

        self.episode_index += 1
        print(f"已处理 Episode {episode_num}，共 {num_frames} 帧")

        # except Exception as e:
