# import torch
# import numpy as np

# def get_ray_maps(intrinsic, c2w, H, W):
#     ###
#     ### intrinsic: vt, 3, 3
#     ### c2w:       vt, 4, 4
#     ### rays:      vt, H, W, 3 and vt, H, W, 3
#     ###
#     vt = intrinsic.shape[0]
#     fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
#     i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()
#     i = i.unsqueeze(0).repeat(vt,1,1)
#     j = j.unsqueeze(0).repeat(vt,1,1)
#     dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     rays_o = c2w[:, :3,-1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
#     viewdir = rays_d/torch.norm(rays_d, dim=-1, keepdim=True)
#     return rays_o, viewdir


import torch


def get_ray_maps(intrinsic, c2w, H, W):
    b, v, t = intrinsic.shape[:3]
    device = intrinsic.device
    dtype = intrinsic.dtype

    fx = intrinsic[..., 0, 0].view(b, v, t, 1, 1)
    fy = intrinsic[..., 1, 1].view(b, v, t, 1, 1)
    cx = intrinsic[..., 0, 2].view(b, v, t, 1, 1)
    cy = intrinsic[..., 1, 2].view(b, v, t, 1, 1)

    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )
    grid_i = i.expand(b, v, t, H, W) + 0.5
    grid_j = j.expand(b, v, t, H, W) + 0.5

    # x = (u - cx) / fx, y = (v - cy) / fy, z = 1
    dirs = torch.stack(
        [(grid_i - cx) / fx, (grid_j - cy) / fy, torch.ones_like(grid_i)], dim=-1
    )

    # rot [b, v, t, 3, 3] -> [b, v, t, 1, 1, 3, 3]
    rot = c2w[..., :3, :3].view(b, v, t, 1, 1, 3, 3)
    # dirs [b, v, t, H, W, 3] -> [b, v, t, H, W, 3, 1]
    rays_d = torch.matmul(rot, dirs.unsqueeze(-1)).squeeze(-1)

    rays_o = c2w[..., :3, 3].view(b, v, t, 1, 1, 3).expand(b, v, t, H, W, 3)

    viewdir = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)

    return rays_o, viewdir


# rays_o, rays_d = get_ray_maps(intrinsic, c2w, h, w)
