from typing import Tuple
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from einops import rearrange

colors = [
    np.array([197, 27, 125]),  # 'pink':
    np.array([215, 48, 39]),  # 'red':
    np.array([252, 141, 89]) - 60,  # 'light_orange':
    np.array([175, 141, 195]),  # 'light_purple':
    np.array([145, 191, 219]),  # 'light_blue':
    np.array([161, 215, 106]) + 20,  # 'light_green':
    np.array([77, 146, 33]) + 20,  # 'green':
    np.array([118, 42, 131]) + 20,  # 'purple':
    np.array([240, 10, 20]),  # red
]


def convert_tensor_to_numpy_array(tensor):
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.numpy()
    return array


def convert_tensor_to_cv_image(tensor):
    tensor = rearrange(tensor, "c h w -> h w c")
    image = convert_tensor_to_numpy_array(tensor)
    image = (image * 255).astype(np.uint8)
    return image


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3, 0:3] = np.matrix(SO)
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3, :]).reshape(1, 12)
    return SE


def get_epipolar(y, P):
    # given relative pose, have function that maps a point in image 1 to another point (or line) in image 2
    # E = [t]×R
    R = P[0, :3, :3]
    t = P[0, :, 3]

    # cross prod rep of t
    t_x = torch.tensor([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]])
    E = (R @ t_x).numpy()

    y_prime = y @ E

    y_prime = y_prime / y_prime[1]  # divide by y coord
    m = -y_prime[0]
    b = -y_prime[2]

    return m, b


def cv_point(point) -> Tuple[int, int]:
    return int(point[0]), int(point[1])


def generate_epipolar_image(src_image, dst_image, rel_pose):
    src_image = src_image.copy()
    dst_image = dst_image.copy()

    radius = 20
    line_width = 10
    epipolar_points_x = 3
    epipolar_points_y = 3

    start_x = -1 + 2 / (epipolar_points_x + 1)
    stop_x = 1
    step_x = 2 / (epipolar_points_x + 1)
    start_y = -1 + 2 / (epipolar_points_y + 1)
    stop_y = 1
    step_y = 2 / (epipolar_points_y + 1)

    # epipolar: dots on img 1
    for y1 in np.arange(start_x, stop_x, step_x):
        for y2 in np.arange(start_y, stop_y, step_y):
            pct_x = (y1 - start_x) / (stop_x - start_x)
            pct_y = (y2 - start_y) / (stop_y - start_y)

            # int((y1 + 0.5) * 2 * 3 + 2 * (y2 + 0.5))
            color_num = int(pct_x * (epipolar_points_x - 1) * epipolar_points_x + pct_y * epipolar_points_y)
            color = (int(colors[color_num][0]), int(colors[color_num][1]), int(colors[color_num][2]))

            y1_img = int((y1 + 1) / 2 * src_image.shape[1])
            y2_img = int((y2 + 1) / 2 * src_image.shape[0])

            cv2.circle(src_image, (y1_img, y2_img), radius, color, -1)

    # epipolar: lines across img 2
    image_epipolar = np.zeros_like(dst_image)

    for y1 in np.arange(start_x, stop_x, step_x):
        for y2 in np.arange(start_y, stop_y, step_y):
            pct_x = (y1 - start_x) / (stop_x - start_x)
            pct_y = (y2 - start_y) / (stop_y - start_y)

            color_num = int(pct_x * (epipolar_points_x - 1) * epipolar_points_x + pct_y * epipolar_points_y)
            color = (int(colors[color_num][0]), int(colors[color_num][1]), int(colors[color_num][2]))

            rot_mtx = pos_quat2SE(rel_pose).reshape([1, 3, 4])
            y = np.array([y1, y2, 1.0], dtype=np.float64)
            m, b = get_epipolar(y, torch.from_numpy(rot_mtx))
            point = np.array([-10.0, -10.0 * m + b])
            point_pa = np.array([10.0, 10.0 * m + b])

            point[0] = (point[0] + 1) / 2 * image_epipolar.shape[1]
            point_pa[0] = (point_pa[0] + 1) / 2 * image_epipolar.shape[1]
            point[1] = (point[1] + 1) / 2 * image_epipolar.shape[0]
            point_pa[1] = (point_pa[1] + 1) / 2 * image_epipolar.shape[0]

            cv2.line(image_epipolar, cv_point(point), cv_point(point_pa), color, line_width)

    dst_image = cv2.addWeighted(image_epipolar, 0.6, dst_image, 0.8, 0)

    out_image = np.concatenate([src_image, dst_image], axis=1)

    return out_image
