from typing import Any, List, Optional
import sys
import os
import time

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

import numpy as np
import hydra.utils
from pytorch_lightning import LightningModule
import pyrootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig
from einops import rearrange
import cv2
from torch import linalg as LA

root = pyrootutils.find_root(__file__)
sys.path.insert(0, str(root / "ctrlc"))
from ctrlc.model.ctrlc_model import GPTran, build_ctrlc

from src.utils.generate_epipolar_image import (generate_epipolar_image,
                                               convert_tensor_to_cv_image,
                                               convert_tensor_to_numpy_array,
                                               )


class CuTiLitModule(LightningModule):
    def __init__(
            self,
            ctrlc: DictConfig,
            ctrlc_checkpoint_path: str,
            transformer: DictConfig,
            vptransformer: DictConfig,
            pos_encoder: DictConfig,
            max_num_line: int,
            hidden_dim: int,
            pool_channels: [int, int],
            pose_regressor_hidden_dim: int,
            pose_size: int,
            criterion: DictConfig,
            test_metric: DictConfig,
            optimizer: DictConfig,
            scheduler: DictConfig,
            matcher: DictConfig,
            vp_criterion: DictConfig,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        assert os.path.exists(ctrlc_checkpoint_path), "ctrlc checkpoint must be existed!"
        ctrlc_checkpoint = torch.load(ctrlc_checkpoint_path)

        self.predictions = {'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}
        self.vp_loss0 = []
        self.vp_loss1 = []
        self.ctrlc: GPTran = build_ctrlc(ctrlc)
        
        self.ctrlc.load_state_dict(ctrlc_checkpoint["model"], strict=False)
        
        self.ctrlc.requires_grad_(False)
        self.ctrlc.eval()

        self.num_image = 2
        self.num_vp = 3
        self.num_head = 8
        self.max_num_line = max_num_line
        self.hidden_dim = hidden_dim

        self.pos_encoder = hydra.utils.instantiate(pos_encoder)

        self.feature_embed = nn.Linear(256, self.hidden_dim)  # 128

        self.image_idx_embedding = nn.Embedding(self.num_image, self.hidden_dim)
        self.line_idx_embedding = nn.Embedding(self.max_num_line, self.hidden_dim)

        self.transformer_block = hydra.utils.instantiate(transformer)
        self.vptransformer_block = hydra.utils.instantiate(vptransformer)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * self.hidden_dim, nhead=self.num_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # translation_regressor_dim = 15 * 20 * 2
        # rotation_regressor_dim = 15 * 20 * 2
        translation_regressor_dim = 15 * 20
        rotation_regressor_dim = 15 * 20
        in_channels = int(self.max_num_line + self.num_vp)
        self.rotation_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.translation_regressor = nn.Sequential(
            nn.Linear(translation_regressor_dim, translation_regressor_dim),
            nn.ReLU(),
            nn.Linear(translation_regressor_dim, translation_regressor_dim),
            nn.ReLU(),
            nn.Linear(translation_regressor_dim, 3),
        )
        self.rotation_regressor = nn.Sequential(
            nn.Linear(rotation_regressor_dim, rotation_regressor_dim),
            nn.ReLU(),
            nn.Linear(rotation_regressor_dim, rotation_regressor_dim),
            nn.ReLU(),
            nn.Linear(rotation_regressor_dim, 4),
        )

        self.image_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.maxpool = nn.MaxPool1d(2, 1)

        self.query_embed = nn.Embedding(2, hidden_dim)

        self.criterion = hydra.utils.instantiate(criterion)
        self.test_camera = hydra.utils.instantiate(test_metric)

        self.matcher = hydra.utils.instantiate(matcher)
        self.vp_criterion = hydra.utils.instantiate(vp_criterion)

        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.automatic_optimization = False

    def forward(self, images: torch.Tensor, lines: torch.Tensor, target):
        endpoint = target['endpoint']
        vps = rearrange(target['vps'], "b i l c -> i b l c ").contiguous()
        batch_size = vps.shape[1]
        
        with torch.no_grad():
            # [bs, line_num, hidden_dim]
            hs0, memory0, pred_view0_vp0, pred_view0_vp1, pred_view0_vp2 = self.ctrlc(
                images[:, 0], lines[:, 0]
            )
            hs1, memory1, pred_view0_vp0, pred_view0_vp1, pred_view0_vp2 = self.ctrlc(
                images[:, 1], lines[:, 1]
            )

        pred_view0_vps = torch.cat([pred_view0_vp0.unsqueeze(1),
                                    pred_view0_vp1.unsqueeze(1),
                                    pred_view0_vp2.unsqueeze(1)], dim=1)
        pred_view1_vps = torch.cat([pred_view0_vp0.unsqueeze(1),
                                    pred_view0_vp1.unsqueeze(1),
                                    pred_view0_vp2.unsqueeze(1)], dim=1)

        index0 = self.matcher(pred_view0_vps, vps[0])
        index1 = self.matcher(pred_view1_vps, vps[1])

        _, tgt_idx0 = self._get_tgt_permutation_idx(index0)
        _, tgt_idx1 = self._get_tgt_permutation_idx(index1)

        tgt_idx0 = tgt_idx0.unsqueeze(0).reshape([batch_size, self.num_vp])
        tgt_idx1 = tgt_idx1.unsqueeze(0).reshape([batch_size, self.num_vp])

        # vp_one_hot0 = F.one_hot(tgt_idx0,num_classes=256).cuda()
        # vp_one_hot1 = F.one_hot(tgt_idx1,num_classes=256).cuda()

        vp0_pos = self.positional_encoding(self.hidden_dim, tgt_idx0).cuda()
        vp1_pos = self.positional_encoding(self.hidden_dim, tgt_idx1).cuda()

        # using last decoder layer's feature
        hs0 = hs0[-1]  # [b x n x c]
        hs1 = hs1[-1]

        hs0_vps, hs0_lines = torch.split(hs0, [3, self.max_num_line], dim=1)
        hs1_vps, hs1_lines = torch.split(hs1, [3, self.max_num_line], dim=1)

        hs0_lines = hs0_lines + self.line_idx_embedding.weight
        hs1_lines = hs1_lines + self.line_idx_embedding.weight

        hs0 = torch.cat([hs0_vps, hs0_lines], dim=1)
        hs1 = torch.cat([hs1_vps, hs1_lines], dim=1)

        feat0 = hs0
        feat1 = hs1

        feat0 = feat0 + self.image_idx_embedding.weight[0]  # + vp0_pos # + vp_one_hot0
        feat1 = feat1 + self.image_idx_embedding.weight[1]  # + vp1_pos # + vp_one_hot1

        # memory0 = memory0.reshape([batch_size, -1, self.hidden_dim])
        # memory1 = memory1.reshape([batch_size, -1, self.hidden_dim])
        memory0 = rearrange(memory0, "b h w c -> b (h w) c").contiguous()
        memory1 = rearrange(memory1, "b h w c -> b (h w) c").contiguous()

        # # 여기부터
        # reshape_feat0 = torch.zeros_like(feat0)
        # reshape_feat1 = torch.zeros_like(feat1)
        # for i in range(feat0.size(0)):
        #     reshape_feat0[i] = feat0[i, tgt_idx0[3*i:3*(i+1)]]
        #     reshape_feat1[i] = feat1[i, tgt_idx1[3*i:3*(i+1)]]
        # feat = torch.cat([reshape_feat0,reshape_feat1],dim=2)
        # feat = self.transformer_encoder(feat)

        # feat0, feat1 = self.transformer_block(feat0, feat1)
        memory0, memory1 = self.transformer_block(memory0, memory1)

        # import pdb; pdb.set_trace()

        # memory0 = memory0.reshape([batch_size, 15, 20, self.hidden_dim])
        # memory1 = memory1.reshape([batch_size, 15, 20, self.hidden_dim])
        memory0 = rearrange(memory0, "b (h w) c -> b c h w", h=15, w=20).contiguous()
        memory1 = rearrange(memory1, "b (h w) c -> b c h w", h=15, w=20).contiguous()

        memory0 = self.image_conv(memory0)
        memory1 = self.image_conv(memory1)
        memory = torch.cat([memory0, memory1], dim=1)

        # feat0, feat1 = self.vptransformer_block(feat0,feat1)

        R = self.rotation_regressor(memory1.reshape([batch_size, -1]))
        T = self.translation_regressor(memory1.reshape([batch_size, -1]))

        pose_preds = torch.cat([T, R], dim=1)

        return {"pred_pose": self.normalize_preds(pose_preds),
                "pred_view0_vps": pred_view0_vps,
                "pred_view1_vps": pred_view1_vps,
                }


    def normalize_preds(self, pred_poses):
        pred_quaternion = pred_poses[:, 3:]
        normalized = torch.norm(pred_quaternion, dim=-1, keepdim=True)
        eps = torch.ones_like(normalized) * .01
        normalize_quat = pred_quaternion / torch.max(normalized, eps)

        return torch.cat([pred_poses[:, :3], normalize_quat], dim=1)

    def positional_encoding(self, d_model, pos):
        pos_enc = torch.zeros((pos.shape[0], pos.shape[1], d_model))
        for i in range(0, d_model, 2):
            pos_enc[:, :, i] = torch.sin(pos / 10000 ** (2 * i / d_model))
            pos_enc[:, :, i + 1] = torch.cos(pos / 10000 ** (2 * i / d_model))
        return pos_enc

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _sqrt_positive_part(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = self._sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)

        return quat_candidates[
               F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
               ].reshape(batch_dim + (4,))

    def eval_camera(self, predictions):
        acc_threshold = {
            "tran": 1.0,
            "rot": 30,
        }
        pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
        pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

        gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
        gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

        top1_error = {
            "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
            "rot": 2 * np.arccos(
                np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
        }
        top1_accuracy = {
            "tran": (top1_error["tran"] < acc_threshold["tran"]).sum()
                    / len(top1_error["tran"]),
            "rot": (top1_error["rot"] < acc_threshold["rot"]).sum()
                   / len(top1_error["rot"]),
        }
        camera_metrics = {
            f"top1 T err < {acc_threshold['tran']}": top1_accuracy["tran"] * 100,
            f"top1 R err < {acc_threshold['rot']}": top1_accuracy["rot"] * 100,
            f"T mean err": np.mean(top1_error["tran"]),
            f"R mean err": np.mean(top1_error["rot"]),
            f"T median err": np.median(top1_error["tran"]),
            f"R median err": np.median(top1_error["rot"]),
        }
        gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:, 0]) * 180 / np.pi}
        pred_mags = {"tran": np.linalg.norm(pred_tran, axis=1), "rot": 2 * np.arccos(pred_rot[:, 0]) * 180 / np.pi}

        os.makedirs("./output2", exist_ok=True)

        tran_graph = np.stack([gt_mags['tran'], pred_mags['tran'], top1_error['tran']], axis=1)
        tran_graph_name = os.path.join('./output2', 'gt_translation_magnitude_vs_error.csv')
        np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

        rot_graph = np.stack([gt_mags['rot'], pred_mags['rot'], top1_error['rot']], axis=1)
        rot_graph_name = os.path.join('./output2', 'gt_rotation_magnitude_vs_error.csv')
        np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
        return camera_metrics

    def on_train_start(self):
        pass

    def shared_step(self, batch: Any):
        images, lines, target = batch
        target_poses = target['poses'][:,1]

        pred_dict = self.forward(images, lines, target)

        vps = rearrange(target['vps'], "b i l c -> i b l c ").contiguous()

        index0 = self.matcher(pred_dict["pred_view0_vps"], vps[0])
        index1 = self.matcher(pred_dict["pred_view1_vps"], vps[1])

        vp_loss0 = self.vp_criterion(pred_dict["pred_view0_vps"], vps[0], index0)
        vp_loss1 = self.vp_criterion(pred_dict["pred_view1_vps"], vps[1], index1)

        pred_poses = pred_dict["pred_pose"]

        loss, loss_dict = self.criterion(target_poses, pred_poses)

        return loss, loss_dict, pred_poses, target_poses

    def training_step(self, batch: Any, batch_idx: int):
        loss, loss_dict, preds, target_pose = self.shared_step(batch)
        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_tr", loss_dict["loss_tr"], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/loss_rot", loss_dict["loss_rot"], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/loss_vp0", loss_dict["loss_vp0"], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/loss_vp1", loss_dict["loss_vp1"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.manual_backward(loss)
        # # loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name,param.grad)

        # optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        # optimizer.step()
        # optimizer.zero_grad()
        # return loss or backpropagation will fail
        return loss

    def train_epoch_end(self, outputs: List[Any]):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.grad)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, loss_dict, preds, targets = self.shared_step(batch)

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_tr", loss_dict["loss_tr"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_rot", loss_dict["loss_rot"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "update_value.csv")

            targets = convert_tensor_to_numpy_array(targets)
            preds = convert_tensor_to_numpy_array(preds)

            file_open_mode = "a" if os.path.exists(output_path) else "w"
            with open(output_path, file_open_mode) as file:
                file.write(f"{self.current_epoch:03d} epoch{',' * 14}\n")
                for target_val, pred_val in zip(targets, preds):
                    file.write(f"{target_val[0]},{target_val[1]},{target_val[2]},"
                               f"{target_val[3]},{target_val[4]},{target_val[5]},{target_val[6]},"
                               f"{pred_val[0]},{pred_val[1]},{pred_val[2]},"
                               f"{pred_val[3]},{pred_val[4]},{pred_val[5]},{pred_val[6]}\n")

        #     images, poses, *rest = batch

        #     src_image = convert_tensor_to_cv_image(images[0, 0])
        #     dst_image = convert_tensor_to_cv_image(images[0, 1])

        #     target_rel_pose = convert_tensor_to_numpy_array(target[0, 1])
        #     pred_rel_pose = convert_tensor_to_numpy_array(preds[0, 1].data)

        #     target_epipolar_image = generate_epipolar_image(src_image, dst_image, target_rel_pose)
        #     pred_epipolar_image = generate_epipolar_image(src_image, dst_image, pred_rel_pose)

        #     epipolar_image = np.concatenate([target_epipolar_image, pred_epipolar_image], axis=0)

        #     # import pdb; pdb.set_trace()

        #     os.makedirs("./output", exist_ok=True)
        #     epipolar_image_path = os.path.join("./output", f"epipolar_{self.current_epoch:03d}.png")
        #     cv2.imwrite(epipolar_image_path, epipolar_image)

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        images, lines, target = batch
        vps = rearrange(target['vps'], "b i l c -> i b l c ").contiguous()

        pred_dict = self.forward(images, lines, target)

        index0 = self.matcher(pred_dict["pred_view0_vps"], vps[0])
        index1 = self.matcher(pred_dict["pred_view1_vps"], vps[1])

        vp_loss0 = self.vp_criterion(pred_dict["pred_view0_vps"], vps[0], index0)
        vp_loss1 = self.vp_criterion(pred_dict["pred_view1_vps"], vps[1], index1)

        self.vp_loss0.append(vp_loss0.tolist())
        self.vp_loss1.append(vp_loss1.tolist())

        predictions = self.test_camera(target['poses'], pred_dict["pred_pose"], self.predictions)

        return predictions, self.vp_loss0, self.vp_loss1

    def test_epoch_end(self, outputs: List[Any]):
        predictions = outputs[0][0]
        vp_loss0 = outputs[0][1]
        vp_loss1 = outputs[0][2]
        print("vp0 average loss : ", sum(vp_loss0) / len(vp_loss0))
        print("vp0 max loss : ", max(vp_loss0))
        print("vp0 min loss : ", min(vp_loss0))
        print("vp1 average loss : ", sum(vp_loss1) / len(vp_loss1))
        print("vp1 max loss : ", max(vp_loss1))
        print("vp1 min loss : ", min(vp_loss1))
        camera_metrics = self.eval_camera(predictions)

        for k in camera_metrics:
            print(k, camera_metrics[k])

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        if self.scheduler is not None:
            scheduler = hydra.utils.instantiate(self.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
