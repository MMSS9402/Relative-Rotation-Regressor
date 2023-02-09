import torch
import torch.nn as nn

from .ctrlc import build_ctrl
from .ctrlc.backbone import build_backbone
from .ctrlc.transformer import build_transformer
from .loftr import build_cuti_module


class CuTi(nn.Module):
    def __init__(self, backbone, ctrlc, transformer, cuti_module, dec_lay):
        super.__init__()

        self.ctrlc1 = ctrlc
        self.ctrlc2 = ctrlc
        self.backbone = backbone
        self.transformer = transformer
        self.cuti_module = cuti_module

        #pos_regressor
        self.layer1 = nn.Linear(dec_lay,1)
        self.layer2 = nn.Linear(dec_lay,1)

    def forward(self, cfg, image1, image2,extra_samples1,extra_samples2):
        # ctrlc model
        hs1 = self.ctrlc1(          #[dec_lay,bs,line_num,hidden_dim]
            image1,
            extra_samples1
            )
        
        hs2 = self.ctrlc2(          #[dec_lay,bs,line_num,hidden_dim]
            image2,
            extra_samples2
        )
        #permute hs1,hs2
        hs1 = hs1.permute(1,2,3,0)  #[bs,line_num,hidden_dim,dec_lay]
        hs2 = hs2.permute(1,2,3,0)  #[bs,line_num,hidden_dim,dec_lay]

        hs1 = self.layer1(hs1)      #[bs,line_num,hidden_dim,1]
        hs2 = self.layer2(hs2)      #[bs,line_num,hidden_dim,1]

        hs1 = hs1.squeeze()         #[bs,line_num,hidden_dim]
        hs2 = hs2.squeeze()         #[bs,line_num,hidden_dim]
        
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
        



def build(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    ctrl1 = build_ctrl(cfg)
    ctrl2 = build_ctrl(cfg)
    cuti_module = build_cuti_module(cfg)

    model = CuTi(ctrl1,ctrl2,cuti_module,) #(dec_lay추가하자)

    return model


