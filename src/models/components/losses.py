from typing import Callable
import torch


class PoseL1Loss(Callable):
    def __init__(self, weights):
        self.weights_tr = weights.translation
        self.weights_rot = weights.rotation

    def __call__(self, pred, target):
        diff = torch.abs(target - pred)

        d_tr, d_rot = diff.split([3, 4], dim=-1)

        loss_tr = d_tr.norm(dim=-1).mean()
        loss_rot = d_rot.norm(dim=-1).mean()
        loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot
        loss_dict = {
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
        }
        return loss, loss_dict
