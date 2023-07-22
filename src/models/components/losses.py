import torch
import torch.nn as nn


class GeodesicLoss(nn.Module):
    def __init__(self, weights):
        self.weights_tr = weights.translation
        self.weights_rot = weights.rotation

    def forward(self, Ps, Gs):
        ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0])

        dP = Ps[:, jj] * Ps[:, ii].inv()
        dG = Gs[0][:, jj] * Gs[0][:, ii].inv()

        d = (dG * dP.inv()).log()

        tau, phi = d.split([3, 3], dim=-1)

        loss_tr = tau.norm(dim=-1).mean()
        loss_rot = phi.norm(dim=-1).mean()
        loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot
        loss_dict = {
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
        }
        return loss, loss_dict
