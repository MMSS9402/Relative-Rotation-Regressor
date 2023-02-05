import torch
import torch.nn as nn

from .ctrlc import build_ctrl
from .ctrlc.backbone import build_backbone
from .ctrlc.transformer import build_transformer
from .loftr import build_cuti_module


class CuTi(nn.Module):
    def __init__(self, backbone, ctrlc, transformer, cuti_module):
        super.__init__()

        self.ctrlc1 = ctrlc
        self.ctrlc2 = ctrlc
        self.backbone = backbone
        self.transformer = transformer
        self.cuti_module = cuti_module

        #pos_regressor
        self.Linear

    def forward(self, cfg, image1, image2,extra_samples1,extra_samples2):
        # ctrlc model
        hs1 = self.ctrlc1(
            image1,
            extra_samples1
            )
        hs2 = self.ctrlc2(
            image2,
            extra_samples2
        )

        # insert hs1,hs2 cuti module
        output = self.cuti_module(
            tgt = hs1,
            memory = hs2,
            tgt_mask = None,
            memory_mask = None,
            tgt_key_padding_mask = None,
            pos = None,
            query_pos = None,
        )

        return output
        
class SetCriterion(nn.Module):
    def __init__(self):
        super.__init__()



def build(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    ctrl1 = build_ctrl(cfg)
    ctrl2 = build_ctrl(cfg)
    cuti_module = build_cuti_module(cfg)

    model = CuTi(ctrl1,ctrl2,cuti_module)

    weight_dict = dict(cfg.LOSS.WEIGHTS)
    losses = cfg.LOSS.LOSSES

    criterion = SetCriterion(
        weight_dict = weight_dict,
        losses=losses,

    )
    criterion.to(device)

    return model, criterion


