from typing import Callable
import torch


class PoseL1Loss(Callable):
    def __init__(self, weights):
        self.weights_tr = weights.translation
        self.weights_rot = weights.rotation

    def __call__(self, pred, target):

        diff = target -  pred

        tau, phi = d.split([3, 3], dim=-1)

        loss_tr = tau.norm(dim=-1).mean()
        loss_rot = phi.norm(dim=-1).mean()
        loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot
        loss_dict = {
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
        }
        return loss, loss_dict
