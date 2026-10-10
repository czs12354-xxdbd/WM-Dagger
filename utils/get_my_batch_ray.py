import torch
import numpy as np
from einops import rearrange


def get_ray_maps_batch(
    intrinsic,
    c2w,
    H=480,
    W=640,
    T=16,
    batch_num=None,
    num_views=None,
    num_timesteps=None,
):
    if isinstance(intrinsic, np.ndarray):
        intrinsic = torch.from_numpy(intrinsic)
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)

    if batch_num is not None:
        intrinsic = intrinsic.unsqueeze(0).repeat(
            batch_num, *([1] * len(intrinsic.shape))
        )
        c2w = c2w.unsqueeze(0).repeat(batch_num, *([1] * len(c2w.shape)))

    if num_views is not None:
        intrinsic = intrinsic.unsqueeze(1).repeat(
            1, num_views, *([1] * (len(intrinsic.shape) - 1))
        )
        c2w = c2w.unsqueeze(1).repeat(1, num_views, *([1] * (len(c2w.shape) - 1)))

    if num_timesteps is not None:
        intrinsic = intrinsic.unsqueeze(2)
        c2w = c2w.unsqueeze(2)
        intrinsic = intrinsic.repeat(
            *([1] * 2), num_timesteps, *([1] * (intrinsic.ndim - 3))
        )
        c2w = c2w.repeat(*([1] * 2), num_timesteps, *([1] * (c2w.ndim - 3)))
    elif intrinsic.ndim in [4, 3]:
        intrinsic = intrinsic.unsqueeze(2)
        c2w = c2w.unsqueeze(2)
        intrinsic = intrinsic.repeat(*([1] * 2), T, *([1] * (intrinsic.ndim - 3)))
        c2w = c2w.repeat(*([1] * 2), T, *([1] * (c2w.ndim - 3)))

    batch_dims = intrinsic.shape[:-4]
    v_dim = intrinsic.shape[-4]
    t_dim = intrinsic.shape[-3]
    device = c2w.device if c2w.device.type != "cpu" else intrinsic.device

    fx = intrinsic[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsic[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsic[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsic[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    i_range = torch.linspace(0.5, W - 0.5, W, device=device)
    j_range = torch.linspace(0.5, H - 0.5, H, device=device)
    i, j = torch.meshgrid(i_range, j_range, indexing="ij")
    i = i.t()
    j = j.t()

    i = i.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
    i = i.repeat(*batch_dims, v_dim, t_dim, 1, 1)
    j = j.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(*batch_dims, v_dim, t_dim, 1, 1)

    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)

    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[..., np.newaxis, np.newaxis, :3, :3], dim=-1
    )

    rays_o = c2w[..., :3, -1].unsqueeze(-2).unsqueeze(-2)
    rays_o = rays_o.expand(*batch_dims, v_dim, t_dim, H, W, 3)

    viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_o, viewdir


if __name__ == "__main__":
    b, v, t = 2, 3, 16
    c2w_np = np.random.randn(b, v, t, 4, 4).astype(np.float32)
    intrinsic_np = np.random.randn(b, v, t, 3, 3).astype(np.float32)
    rays_o, viewdir = get_ray_maps_batch(intrinsic_np, c2w_np, H=480, W=640)
    print("动态相机光线原点形状:", rays_o.shape)

    c2w_static_np = np.random.randn(b, v, 4, 4).astype(np.float32)
    intrinsic_static_np = np.random.randn(b, v, 3, 3).astype(np.float32)
    rays_o_static, viewdir_static = get_ray_maps_batch(
        intrinsic=intrinsic_static_np, c2w=c2w_static_np, H=480, W=640, T=10
    )
    print("固定相机光线原点形状:", rays_o_static.shape)

    c2w_single_batch = np.random.randn(v, 4, 4).astype(np.float32)
    intrinsic_single_batch = np.random.randn(v, 3, 3).astype(np.float32)
    rays_o_single, viewdir_single = get_ray_maps_batch(
        intrinsic=intrinsic_single_batch, c2w=c2w_single_batch, H=480, W=640, T=5
    )
    print("单batch光线原点形状:", rays_o_single.shape)
