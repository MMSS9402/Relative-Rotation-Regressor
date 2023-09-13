import os
import json

import numpy as np

from src.datamodule.components.base import RGBDDataset


class MatterportDataset(RGBDDataset):
    # scale depth to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(
            self,
            data_path: str,
            ann_filename: str,
            reshape_size: (int, int) = (480, 640),
            use_mini_dataset: bool = False,
    ):
        super(MatterportDataset, self).__init__(
            name="Matterport",
            data_path=data_path,
            ann_filename=ann_filename,
            reshape_size=reshape_size,
            use_mini_dataset=use_mini_dataset,
        )

    def _build_dataset(self):
        base_pose = np.array([0, 0, 0, 0, 0, 0, 1])

        with open(os.path.join(self.data_path, "mp3d_planercnn_json", self.ann_filename)) as file:
            split = json.load(file)

        images_list = []
        lines_list = []
        vps_list = []
        poses_list = []
        intrinsics_list = []

        original_basepath = "/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20"

        for data in split["data"].values():
            vps = []
            images = []
            lines = []
            for img_idx in ["0", "1"]:
                img_path = data[img_idx]["file_name"].replace(original_basepath, self.data_path)
                line_path = img_path.replace(".png", "_line.csv",)

                vp1 = data[img_idx]['vp1']
                vp2 = data[img_idx]['vp2']
                vp3 = data[img_idx]['vp3']

                gt_vps = np.array([vp1, vp2, vp3])
                vps.append(gt_vps)

                images.append(img_path)
                lines.append(line_path)

            rel_pose = np.array(data["rel_pose"]["position"] + data["rel_pose"]["rotation"])

            # on matterport, we scale depths to balance rot & trans loss
            rel_pose[:3] /= self.DEPTH_SCALE

            # swap 3 & 6, we want W last for consistency with our other datasets
            rel_pose[6], rel_pose[3] = rel_pose[3], rel_pose[6]

            # normalize quaternions to have positive "W"
            if rel_pose[6] < 0:
                rel_pose[3:] *= -1
            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array(
                [[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]]
            )  # 480 x 640 imgs

            images_list.append(images)
            lines_list.append(lines)
            vps_list.append(vps)
            poses_list.append(poses)
            intrinsics_list.append(intrinsics)

        scene_info = {
            "images": images_list,
            "poses": poses_list,
            "intrinsics": intrinsics_list,
            "lines": lines_list,
            'vps': vps_list,
        }
        return scene_info