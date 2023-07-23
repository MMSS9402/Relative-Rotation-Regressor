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

        images = []
        lines = []
        vps = []
        poses = []
        intrinsics = []

        for data in split["data"]:

            for img_idx in ["0", "1"]:
                img_name = os.path.join(self.data_path,
                                        "/".join(data[img_idx]["file_name"].split("/")[6:]))

                line_name = img_name.split("/")
                line_name[9] = img_name.split("/")[9].split(".")[0] + "_line.csv"
                line_name = "/".join(line_name)

                vp1 = data[img_idx]['vp1']
                vp2 = data[img_idx]['vp2']
                vp3 = data[img_idx]['vp3']

                gt_vps = np.array([vp1, vp2, vp3])
                vps.append(gt_vps)

                images.append(img_name)
                lines.append(line_name)

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

        scene_info = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "lines": lines,
            'vps': vps,
        }
        return scene_info