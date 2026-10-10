import numpy as np
import cv2
import matplotlib.cm as cm
import torch
import os
from datetime import datetime
from einops import rearrange


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)  # r=w, i=x, j=y, k=z
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_transformation_matrix_from_quat(quat):
    rot_quat = quat[..., 3:]
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[..., :3]

    output = torch.eye(4, device=quat.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    output = output.repeat(quat.shape[0], quat.shape[1], 1, 1)  # (b, t, 4, 4)

    output[..., :3, :3] = rot
    output[..., :3, 3] = trans
    return output


def simple_radius_gen_func(xyzs, c_xyzs):
    radius = (
        torch.clamp(
            4 - torch.sqrt(((xyzs - c_xyzs.unsqueeze(1)) ** 2).sum(-1)), min=0, max=1
        )
        * 100
    )
    return radius


def get_batch_arm_traj_maps(
    pose,
    sample_size=(480, 640),
    intrinsic=None,
    w2c=None,
    c2w=None,
    radius_gen_func=None,
    save_dir=None,
):
    if intrinsic is None:
        intrinsic = np.array([[553.4, 0, 320], [0, 565.2, 240], [0, 0, 1]])
    if c2w is None:
        c2w = np.array(
            [
                [0.0000000, 0.5735659, -0.8191594, 3.5],
                [1.0000000, 0.0000000, 0.0000000, 0.5],
                [0.0000000, -0.8191594, -0.5735659, 2.09],
                [0.0000000, 0.0000000, 0.0000000, 1],
            ]
        )
    if w2c is None:
        w2c = np.linalg.inv(c2w)

    h, w = sample_size
    colormap = cm.Greens
    color_list = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    assert (
        pose.ndim == 3 and pose.shape[-1] == 8
    ), f"pose应具有形状 (b, t, 8)，实际为 {pose.shape}"
    b, t, _ = pose.shape
    device = pose.device

    ee_key_pts = (
        torch.tensor(
            [[0, 0, 0, 1], [0.3, 0, 0, 1], [0, 0.3, 0, 1], [0, 0, 0.3, 1]],
            dtype=torch.float32,
            device=device,
        )
        .view(1, 1, 1, 4, 4)
        .permute(0, 1, 2, 4, 3)
    )  # (1,1,1,4,4)

    arm_pose_mat = get_transformation_matrix_from_quat(pose[..., 0:7])  # (b, t, 4, 4)

    if isinstance(w2c, np.ndarray):
        w2c = torch.tensor(w2c, dtype=torch.float32, device=device)
    if w2c.ndim == 2:
        w2c = w2c.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    assert w2c.shape in [
        (b, 1, 4, 4),
        (1, 1, 4, 4),
    ], f"w2c形状应为 (b,1,4,4) 或 (1,1,4,4)，实际为 {w2c.shape}"
    ee2cam = torch.matmul(w2c, arm_pose_mat)  # (b, t, 4, 4)

    pts = torch.matmul(ee2cam.unsqueeze(2), ee_key_pts)
    pts = pts.squeeze(2)  # (b, t, 4, 4)

    if isinstance(intrinsic, np.ndarray):
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    assert intrinsic.shape in [
        (b, 1, 3, 3),
        (1, 1, 3, 3),
    ], f"intrinsic形状应为 (b,1,3,3) 或 (1,1,3,3)，实际为 {intrinsic.shape}"

    uvs0 = torch.matmul(intrinsic, pts[..., :3, :])
    uvs = (
        (uvs0 / pts[..., 2:3, :])[..., :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)
    )  # (b, t, 4, 2)

    if isinstance(c2w, np.ndarray):
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)  # (1, 4, 4)
    assert c2w.shape in [
        (b, 4, 4),
        (1, 4, 4),
    ], f"c2w形状应为 (b,4,4) 或 (1,4,4)，实际为 {c2w.shape}"
    cam_xyz = c2w[..., :3, 3]

    arm_xyz = pose[..., 0:3]  # (b, t, 3)
    if radius_gen_func is None:
        arm_dist = torch.full((b, t), 50, device=device, dtype=torch.float32)
    else:
        arm_dist = radius_gen_func(arm_xyz, cam_xyz)  # (b, t)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(base_save_dir, exist_ok=True)

    batch_img_list = []
    for batch_idx in range(b):
        sample_save_dir = (
            os.path.join(base_save_dir, f"sample_{batch_idx:03d}")
            if (save_dir is not None)
            else None
        )
        if sample_save_dir is not None:
            os.makedirs(sample_save_dir, exist_ok=True)

        time_img_list = []
        for time_idx in range(t):
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            normalized_value = pose[batch_idx, time_idx, 7].item() / 1
            color = colormap(normalized_value)[:3]
            color = tuple(int(c * 255) for c in color)

            points = uvs[batch_idx, time_idx]  # (4, 2)
            base = points[0].cpu().numpy()
            # print(base)
            if 0 <= base[0] < w and 0 <= base[1] < h:
                center = tuple(base[:2])
                radius = int(arm_dist[batch_idx, time_idx].item())
                # print('radius:',radius)
                # print('center:',center)
                # print('color:',color)
                cv2.circle(img, center, radius, color, -1)

                for dir_idx, point in enumerate(points[1:]):
                    end_point = point[:2].cpu().numpy()
                    if 0 <= end_point[0] < w and 0 <= end_point[1] < h:
                        cv2.line(img, center, tuple(end_point), color_list[dir_idx], 8)

            if sample_save_dir is not None:
                img_path = os.path.join(sample_save_dir, f"frame_{time_idx:04d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            time_img_list.append(img / 255.0)

        time_img_list = np.stack(time_img_list, axis=0)
        batch_img_list.append(time_img_list)

    batch_img_array = np.stack(batch_img_list, axis=0)  # (b, t, h, w, 3)
    batch_img_tensor = torch.from_numpy(batch_img_array).float()

    batch_img_tensor = rearrange(batch_img_tensor, "b t h w c -> b c 1 t h w")

    batch_img_tensor = batch_img_tensor.to(device)

    return batch_img_tensor


if __name__ == "__main__":
    batch_pose = np.array(
        [
            [
                [0.8, 1.0, 1.0, 0, 0, 1, 0, 0],
                [0.8, 1.2, 1.2, 0, 0, 1, 0, 1],
                [0.8, 1.4, 1.4, 0, 0, 1, 0, 1],
            ],
            [
                [1.0, 1.0, 1.0, 0, 0, 1, 0, 0],
                [1.0, 1.2, 1.2, 0, 0, 1, 0, 1],
                [1.0, 1.4, 1.4, 0, 0, 1, 0, 1],
            ],
        ]
    )

    result = get_batch_arm_traj_maps(
        pose=batch_pose,
        save_dir="./batch_traj_imgs",
        radius_gen_func=simple_radius_gen_func,
    )

    print(f"输出张量形状: {result.shape}")
