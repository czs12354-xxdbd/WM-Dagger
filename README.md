# WM-DAgger

**WM-DAgger: Enabling Efficient Data Aggregation for Imitation Learning with World Models**

WM-DAgger uses a world model to generate recovery trajectories for robotic imitation learning. It includes training, inference, and data aggregation code.

## Setup

This project targets Python 3.10 and a CUDA GPU. Create an environment and install the pinned dependencies:

```bash
conda create -n wm_dagger python=3.10.4
conda activate wm_dagger
pip install -r requirements.txt
```

Download the required dataset and model checkpoints separately. They are not included in this repository:

- [Dataset](https://drive.google.com/file/d/1n6I1y7zYbw2ML6MJ8Pzl_qjObdvqzQv2/view?usp=sharing)
- [GE-Sim checkpoint](https://modelscope.cn/models/agibot_world/Genie-Envisioner/file/view/master/ge_sim_cosmos_v0.1.safetensors)
- [Cosmos Predict2 2B components](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)

Before running, update the dataset roots and checkpoint paths in the selected YAML configuration. The example configurations contain paths from the development environment.

## Run

```bash
bash scripts/train.sh main.py configs/cosmos_model/acwm_cosmos_train.yaml
bash scripts/infer.sh main.py configs/cosmos_model/acwm_cosmos_infer.yaml
bash scripts/infer.sh main_aug_lerobot_filering.py configs/cosmos_model/acwm_cosmos_infer.yaml
```

For a single GPU, set `CUDA_VISIBLE_DEVICES=0` before running a script. Multi-node execution uses `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

The optional DMD code also requires `DMD_ROOT` to point to a local `dmd_diffusion` checkout containing its source and checkpoints.

## Background and attribution

WM-DAgger builds on [Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner-V1). See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for the upstream license terms and component attribution.
