import torch
import torch.nn as nn
from einops.einops import rearrange

from .ctrlc import build_ctrl
from .ctrlc import build_backbone
from .loftr import LocalFeatureTransformer

class CuTi(nn.Module):
    def __init__(self):
        super.__init__()

        self.ctrlc1 = ctrlc
        self.ctrlc2 = ctrlc
        self.backbone = backbone
        self.loftr = LocalFeatureTransformer()

    def forward(self):
        #ctrlc model
        hs1 = self.ctrlc1(backbone,
        transformer,
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
        )
        hs2 = self.ctrlc2(backbone,
        transformer,
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
        )
    # insert h1,h2 cuti module


'''
class SetCriterion(nn.Module):
    def __init__(self):
        super.__init__()
'''

'''
def build(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    ctrl1 = build_ctrl(cfg)
    ctrl2 = build_ctrl(cfg)
    cuti_module = build_cuti_module

    model = CuTi(ctrl1,ctrl2,cuti_module)

    weight_dict = dict(cfg.LOSS.WEIGHTS)
    losses = cfg.LOSS.LOSSES

    criterion = SetCriterion(
        weight_dict = weight_dict,
        losses=losses,

    )
    criterion.to(device)

    return model, criterion

        
