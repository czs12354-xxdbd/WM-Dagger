import numpy as np
import cv2
import matplotlib.cm as cm
import torch
from einops import rearrange


# def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
#     r, i, j, k = torch.unbind(quaternions, -1)  # r=w, i=x, j=y, k=z
#     two_s = 2.0 / (quaternions * quaternions).sum(-1)
#     o = torch.stack(
#         (
#             1 - two_s * (j * j + k * k),
#             two_s * (i * j - k * r),
#             two_s * (i * k + j * r),
#             two_s * (i * j + k * r),
#             1 - two_s * (i * i + k * k),
#             two_s * (j * k - i * r),
#             two_s * (i * k - j * r),
#             two_s * (j * k + i * r),
#             1 - two_s * (i * i + j * j),
#         ),
#         -1,
#     )
#     return o.reshape(quaternions.shape[:-1] + (3, 3))


# def get_transformation_matrix_from_quat(quat):

#     """
#     output = torch.eye(4).unsqueeze(0).repeat(quat.shape[0], 1, 1)
#     return output


# def simple_radius_gen_func(xyzs, c_xyzs):
#     radius = torch.clamp(
#         1.0 - torch.sqrt(((xyzs - c_xyzs) **2).sum(-1)) - 0.07 / (0.8 - 0.07),
#         min=0,
#         max=1
#     ) * 100
#     return radius


# def get_single_arm_traj_maps(pose, w2c, c2w, intrinsic, sample_size, radius_gen_func=None):
#     """

#     """
#     h, w = sample_size

#     if isinstance(pose, np.ndarray):
#         pose = torch.tensor(pose, dtype=torch.float32)

#     ee_key_pts = torch.tensor([
#     ], dtype=torch.float32, device=pose.device).view(1, 1, 4, 4).permute(0, 1, 3, 2)

#     arm_pose_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(dim=0)

#     ee2cam = torch.matmul(w2c, arm_pose_mat)

#     correct_matrix = torch.tensor([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0.23],
#         [0, 0, 0, 1]
#     ], dtype=torch.float32, device=pose.device).view(1, 1, 4, 4)
#     ee2cam = torch.matmul(ee2cam, correct_matrix)

#     pts = torch.matmul(ee2cam, ee_key_pts)

#     intrinsic = intrinsic.unsqueeze(0).unsqueeze(0)


#     all_img_list = []
#     icam = 0
#     arm_xyz = pose[:, 0:3].clone()
#     cam_xyz = c2w[icam, :, :3, 3].clone()

#     if radius_gen_func is None:
#     else:
#         arm_dist = radius_gen_func(arm_xyz, cam_xyz)

#     img_list = []
#         img = np.zeros((h, w, 3), dtype=np.uint8) + 50

#         color = colormap(normalized_value)[:3]
#         color = tuple(int(c * 255) for c in color)

#         points = uvs[icam, i]

#         if 0 <= base[0] < w and 0 <= base[1] < h:
#             center = tuple(base[:2])
#             radius = int(arm_dist[i]) if radius_gen_func else arm_dist

#                 end_point = np.array(point[:2])
#                 if 0 <= end_point[0] < w and 0 <= end_point[1] < h:

#         img_list.append(img / 255.)

#     img_list = np.stack(img_list, axis=0)

#     all_img_list = np.stack(all_img_list, axis=0)
#     all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()

#     return all_img_list


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Copied from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L43C1-L72C54

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
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
    ### quat: (b, 7)
    rot_quat = quat[:, 3:]
    rot_quat = rot_quat[:, [3, 0, 1, 2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


def simple_radius_gen_func(xyzs, c_xyzs):
    ### A simple emperical function to generate raidus based on the distances between end-effectors and the camera
    radius = (
        torch.clamp(
            1.0 - torch.sqrt(((xyzs - c_xyzs) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07),
            min=0,
            max=1,
        )
        * 100
    )
    return radius


def get_traj_maps(pose, w2c, c2w, intrinsic, sample_size, radius_gen_func=None):

    h, w = sample_size
    colormap_l = cm.Greens
    colormap_r = cm.Reds
    color_list_l = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
    color_list_r = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)

    ee_key_pts = (
        torch.tensor(
            [[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]],
            dtype=torch.float32,
            device=pose.device,
        )
        .view(1, 1, 4, 4)
        .permute(0, 1, 3, 2)
    )

    ### 1, t, 4, 4
    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(dim=0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(dim=0)

    ### v, t, 4, 4
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    correct_matrix = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.23], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=pose.device,
    ).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, correct_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, correct_matrix)

    ### v, t, 4, 4
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)

    ### v, 1, 3, 3
    intrinsic = intrinsic.unsqueeze(1)

    ### v, t, 3, 4
    uvs_l0 = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (
        (uvs_l0 / pts_l[:, :, 2:3, :])[:, :, :2, :]
        .permute(0, 1, 3, 2)
        .to(dtype=torch.int64)
    )

    ### v, t, 3, 4
    uvs_r0 = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (
        (uvs_r0 / pts_r[:, :, 2:3, :])[:, :, :2, :]
        .permute(0, 1, 3, 2)
        .to(dtype=torch.int64)
    )

    all_img_list = []

    for icam in range(w2c.shape[0]):

        l_xyz = pose[:, 0:3].clone()
        r_xyz = pose[:, 8:11].clone()
        c_xyz = c2w[icam, :, :3, 3].clone()

        if radius_gen_func is None:
            l_dist = 50
            r_dist = 50
        else:
            l_dist = radius_gen_func(l_xyz, c_xyz)
            r_dist = radius_gen_func(r_xyz, c_xyz)

        img_list = []
        for i in range(pose.shape[0]):

            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = colormap_l(normalized_value_l)[:3]  # Get RGB values
            color_r = colormap_r(normalized_value_r)[:3]  # Get RGB values
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            i_coord_list = []
            for points, color, colors, radius, lr_tag, eef in zip(
                [uvs_l[icam, i], uvs_r[icam, i]],
                [color_l, color_r],
                [color_list_l, color_list_r],
                [l_dist[i], r_dist[i]],
                ["left", "right"],
                [normalized_value_l, normalized_value_r],
            ):
                base = np.array(points[0])  # points:[4,3]
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                point = np.array(points[0][:2])
                radius = int(radius)
                cv2.circle(img, tuple(point), radius, color, -1)
                # color_circle = int(128*eef)+128
                # cv2.circle(img, tuple(point), radius, (color_circle, color_circle, color_circle), 10)

            for points, color, colors, lr_tag in zip(
                [uvs_l[icam, i], uvs_r[icam, i]],
                [color_l, color_r],
                [color_list_l, color_list_r],
                ["left", "right"],
            ):
                base = np.array(points[0])  # points:[4,3]
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                for i, point in enumerate(points):
                    point = np.array(point[:2])
                    if i == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[i - 1], 8)

            img_list.append(img / 255.0)

        img_list = np.stack(img_list, axis=0)  ### t,h,w,c
        all_img_list.append(img_list)

    all_img_list = np.stack(all_img_list, axis=0)  ### ncam, t, h, w, c
    all_img_list = rearrange(
        torch.tensor(all_img_list), "v t h w c -> c v t h w"
    ).float()

    return all_img_list
