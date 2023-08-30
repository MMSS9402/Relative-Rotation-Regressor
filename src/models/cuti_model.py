from typing import Any, List
import sys
import os
import numpy as np
import hydra.utils
from pytorch_lightning import LightningModule
import pyrootutils
import torch
import torch.nn as nn
from lietorch import SE3
from omegaconf import DictConfig
from einops import rearrange

root = pyrootutils.find_root(__file__)
sys.path.insert(0, str(root / "ctrlc"))
from ctrlc.model.ctrlc_model import GPTran, build_ctrlc


class CuTiLitModule(LightningModule):
    def __init__(
            self,
            ctrlc: DictConfig,
            ctrlc_checkpoint_path: str,
            transformer: DictConfig,
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
            Rel_criterion: DictConfig,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # assert os.path.exists(ctrlc_checkpoint_path), "ctrlc checkpoint must be exists!"
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
        self.max_num_line = max_num_line
        self.hidden_dim = hidden_dim

        self.pos_encoder = hydra.utils.instantiate(pos_encoder)

        self.feature_embed = nn.Linear(256, self.hidden_dim) # 128

        self.image_idx_embedding = nn.Embedding(self.num_image, self.hidden_dim)
        self.line_idx_embedding = nn.Embedding(3, self.hidden_dim)

        self.transformer_block = hydra.utils.instantiate(transformer)

        self.pool_transformer_output = nn.Sequential(
            nn.Conv2d(hidden_dim, pool_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_channels[0]),
            nn.ReLU(),
            nn.Conv2d(pool_channels[0], pool_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_channels[1])
        )

        pose_regressor_input_dim = int(self.num_image * pool_channels[1] * (3))

        self.pose_regressor = nn.Sequential(
            nn.Linear(pose_regressor_input_dim, pose_regressor_hidden_dim),
            nn.ReLU(),
            nn.Linear(pose_regressor_hidden_dim, pose_regressor_hidden_dim),
            nn.ReLU(),
            nn.Linear(pose_regressor_hidden_dim, self.num_image * pose_size),
            nn.Unflatten(1, (self.num_image, pose_size)),
        )

        self.criterion = hydra.utils.instantiate(criterion)
        self.test_camera = hydra.utils.instantiate(test_metric)
        
        self.matcher = hydra.utils.instantiate(matcher)
        self.vp_criterion = hydra.utils.instantiate(vp_criterion)
        self.Rel_criterion = hydra.utils.instantiate(Rel_criterion)

        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, images: torch.Tensor, lines: torch.Tensor,endpoint: torch.Tensor):

        endpoint[0] = torch.tensor(endpoint[0],dtype=torch.float32)
        endpoint[1] = torch.tensor(endpoint[1],dtype=torch.float32)
        # print("lines",lines[0])
        # [dec_layer, bs, line_num, hidden_dim]
        hs0, memory0, pred0_vp0, pred0_vp1, pred0_vp2 = self.ctrlc(
            images[:, 0],
            lines[:, 0]
        )
        hs1, memory1, pred1_vp0, pred1_vp1, pred1_vp2 = self.ctrlc(
            images[:, 1],
            lines[:, 1]
        )
        pred0_vp = torch.cat([torch.cat([pred0_vp0.unsqueeze(1),pred0_vp1.unsqueeze(1)],dim=1),pred0_vp2.unsqueeze(1)],dim=1)
        pred1_vp = torch.cat([torch.cat([pred1_vp0.unsqueeze(1),pred1_vp1.unsqueeze(1)],dim=1),pred1_vp2.unsqueeze(1)],dim=1)
        # hs0 = rearrange(hs0, "d b n c -> b n c d").contiguous()
        # hs1 = rearrange(hs1, "d b n c -> b n c d").contiguous()
        # using last decoder layer's feature
        hs0 = hs0[-1]  # [b x n x c]
        hs1 = hs1[-1]

        batch_size = hs0.shape[0]
        #print(batch_size)

        # hs0 = self.feature_embed(hs0)
        # hs1 = self.feature_embed(hs1)
        hs0 = hs0 + self.image_idx_embedding.weight[0] + self.line_idx_embedding.weight
        hs1 = hs1 + self.image_idx_embedding.weight[1] + self.line_idx_embedding.weight
        
        # feat0 = torch.cat([hs0 ], dim=1)
        # feat1 = torch.cat([hs1 ], dim=1)
        feat0 = hs0
        feat1 = hs1
        feat0, feat1 = self.transformer_block(feat0, feat1)

        feat = torch.cat([feat0.unsqueeze(0), feat0.unsqueeze(0)], dim=0)
        feat = feat.reshape([batch_size, self.num_image, 3, -1])
        feat = rearrange(feat, "b i l c -> b c i l").contiguous()

        pooled_feat = self.pool_transformer_output(feat)
        pose_preds = self.pose_regressor(pooled_feat.reshape([batch_size, -1]))
        
        #print(pose_preds)
#self.normalize_preds(pose_preds)
        return self.normalize_preds(pose_preds), pred0_vp, pred1_vp

    def normalize_preds(self, pose_preds):
        pred_out_Gs = SE3(pose_preds)
        #print("pose_pred",pose_preds.data)
        Gs = SE3.IdentityLike(pred_out_Gs)
        #print("gs",Gs.data.shape)
        normalized = pred_out_Gs.data[:, :, 3:].norm(dim=-1).unsqueeze(2)
        #print(normalized)
        eps = torch.ones_like(normalized) * .01
        pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
        pred_out_Gs_new.data[:, :, 3:] = pred_out_Gs.data[:, :, 3:] / torch.max(normalized, eps)

        these_out_Gs = SE3(torch.cat([Gs[:, :1].data, pred_out_Gs_new.data[:, 1:]], dim=1))
        out_Gs = [these_out_Gs]
        return out_Gs
    

    def eval_camera(self,predictions):
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
            "rot": 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
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
        gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:,0]) * 180 / np.pi}
        pred_mags = {"tran":np.linalg.norm(pred_tran, axis=1), "rot": 2 * np.arccos(pred_rot[:,0]) * 180 / np.pi}
        
        tran_graph = np.stack([gt_mags['tran'], pred_mags['tran'],top1_error['tran']],axis=1)
        tran_graph_name = os.path.join('/home/kmuvcl/source/oldCuTi/CuTi/logs/output2', 'gt_translation_magnitude_vs_error.csv')
        np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

        rot_graph = np.stack([gt_mags['rot'],pred_mags['rot'], top1_error['rot']],axis=1)
        rot_graph_name = os.path.join('/home/kmuvcl/source/oldCuTi/CuTi/logs/output2', 'gt_rotation_magnitude_vs_error.csv')
        np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
        return camera_metrics

    def on_train_start(self):
        pass
    
    def shared_step(self, batch: Any):
        images, poses, intrinsics, lines, vps, endpoint = batch
        Ps = SE3(poses)
        
        pose_preds,pred0_vp,pred1_vp = self.forward(images, lines, endpoint)
    
        loss, loss_dict = self.criterion(Ps, pose_preds)
        
        # loss, loss_dict = self.Rel_criterion(poses, pose_preds)

        return loss, loss_dict, pose_preds, poses

    def training_step(self, batch: Any, batch_idx: int):
        loss, loss_dict, preds, targets = self.shared_step(batch)

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def train_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, loss_dict, preds, targets = self.shared_step(batch)

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        
        images, poses, intrinsics, lines, vps, endpoint = batch
        vps = rearrange(vps, "b i l c -> i b l c ").contiguous()
    
        pose_preds,pred0_vp,pred1_vp = self.forward(images, lines, endpoint)
        
        index0 = self.matcher(pred0_vp,vps[0])
        index1 = self.matcher(pred1_vp,vps[1])

        vp_loss0 = self.vp_criterion(pred0_vp,vps[0],index0)
        vp_loss1 = self.vp_criterion(pred1_vp,vps[1],index1)
        
        self.vp_loss0.append(vp_loss0.tolist())
        self.vp_loss1.append(vp_loss1.tolist())

        predictions = self.test_camera(poses, pose_preds,self.predictions)

        return predictions,self.vp_loss0,self.vp_loss1

    def test_epoch_end(self, outputs: List[Any]):
        predictions = outputs[0][0]
        vp_loss0 = outputs[0][1]
        vp_loss1 = outputs[0][2]
        print("vploss0 average loss : ",sum(vp_loss0)/len(vp_loss0))
        print("vploss0 max loss : ",max(vp_loss0))
        print("vploss0 min loss : ",min(vp_loss0))
        print("vploss1 average loss : ",sum(vp_loss1)/len(vp_loss1))
        print("vploss1 max loss : ",max(vp_loss1))
        print("vploss1 min loss : ",min(vp_loss1))
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
