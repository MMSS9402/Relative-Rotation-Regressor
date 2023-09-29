from typing import Callable
import torch


class L1Loss(Callable):
    def __init__(self, weights):
        self.weights_tr = weights.translation
        self.weights_rot = weights.rotation
        self.weights_vp = weights.vp

    # def __call__(self, Ps, Gs):
    def __call__(self, target, pred, vp_loss0=None,vp_loss1=None):
        # d_target = target[:, 1]
        # d_pred = pred[:, 1]
        # d_pred = pred
        diff = torch.abs(pred - target)
        diff_tr, diff_rot = diff.split([3, 4], dim=-1)
        loss_tr = diff_tr.norm(dim=-1).mean()
        loss_rot = diff_rot.norm(dim=-1).mean()

        if vp_loss0 is not None and vp_loss1 is not None:
            loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot + self.weights_vp * vp_loss0 + self.weights_vp * vp_loss1
            loss_dict = {
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
            "loss_vp0" : vp_loss0,
            "loss_vp1" : vp_loss1
        }
        else:
            loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot
            loss_dict = {
                "loss_tr": loss_tr,
                "loss_rot": loss_rot,
            }
        return loss, loss_dict
