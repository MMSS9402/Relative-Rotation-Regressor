from ctrlc import transformer
import torch
import torch.nn as nn
import lietorch
from lietorch import SE3

from ctrlc import build_ctrl
from ctrlc.backbone import build_backbone
from ctrlc.transformer import build_transformer
from loftr import build_cuti_module


class CuTi(nn.Module):
    def __init__(self, backbone, ctrlc, transformer, cuti_module, decoder_layer):
        super().__init__()
        self.pose_size = 7
        self.num_images = 2
        
        self.ctrlc1 = ctrlc
        self.ctrlc2 = ctrlc
        self.backbone = backbone
        self.transformer = transformer
        self.cuti_module = cuti_module
        self.decoder_layer = decoder_layer

        #demension check
        self.layer1 = nn.Linear(self.decoder_layer,1)
        self.layer2 = nn.Linear(self.decoder_layer,1)
        
        #1/2 layer
        self.layer3 = nn.Linear(2,1)

        #dimension
        self.H = 32768
        self.H2 = 8192
        self.H3 = 2048
        self.H4 = 512
        
        #pos_regressor
        self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2),
                nn.ReLU(),
                nn.Linear(self.H2, self.H3),
                nn.ReLU(),
                nn.Linear(self.H3, self.H4),
                nn.ReLU(), 
                nn.Linear(self.H4, self.num_images * self.pose_size),
                nn.Unflatten(1, (self.num_images, self.pose_size))
            )
    def normalize_preds(self, Gs, pose_preds):
        pred_out_Gs = SE3(pose_preds)
        
        normalized = pred_out_Gs.data[:,:,3:].norm(dim=-1).unsqueeze(2)
        eps = torch.ones_like(normalized) * .01
        pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
        pred_out_Gs_new.data[:,:,3:] = pred_out_Gs.data[:,:,3:] / torch.max(normalized, eps)
        
        # print("shape____1",Gs[:,:1].data.shape)
        # print("shape____2",pred_out_Gs_new.data[:,1:].shape)
        
        these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs_new.data[:,1:]], dim=1))
            
        # if inference:
        #     out_Gs = these_out_Gs.data[0].cpu().numpy()
        # else:
        out_Gs = [these_out_Gs]

        return out_Gs

    def forward(self, images, lines, Gs):#수정 필요
        # ctrlc model
        #print(images.shape)
        lines = lines.permute(1,0,2,3)
        images = images.permute(1,0,2,3,4)
        hs1 = self.ctrlc1(          #[dec_lay,bs,line_num,hidden_dim]
            images[0],
            lines[0]
            )
        
        hs2 = self.ctrlc2(          #[dec_lay,bs,line_num,hidden_dim]
            images[1],
            lines[1]
        )
        #permute hs1,hs2
        hs1 = hs1.permute(1,2,3,0)  #[bs,line_num,hidden_dim,dec_lay]
        hs2 = hs2.permute(1,2,3,0)  #[bs,line_num,hidden_dim,dec_lay]

        hs1 = self.layer1(hs1)      #[bs,line_num,hidden_dim,1]
        hs2 = self.layer2(hs2)      #[bs,line_num,hidden_dim,1]

        hs1 = hs1.squeeze(3)         #[bs,line_num,hidden_dim]
        hs2 = hs2.squeeze(3)         #[bs,line_num,hidden_dim]
        
        # insert hs1,hs2 cuti module
        hs1 , hs2 = self.cuti_module(hs1,hs2,None,None)
        
        #print("hs1______",hs1.shape)
        output =  torch.cat([hs1,hs2], dim=0)
        #print("output::::____",output.shape)
        output = output.permute(1,2,0)
        output = self.layer3(output)
        output = output.permute(2,0,1)
        #output = output.squeeze()
        Batch_size, _, _ = output.shape
        output = output.reshape([Batch_size,-1])
        #print("flatten________",output.shape)
        
        pose_preds = self.pose_regressor(output)
        #print("Pose_pred______",pose_preds.shape)
        #print("GS__________",Gs.shape)
        

        return self.normalize_preds(Gs, pose_preds)
        



def build(cfg):
    device = torch.device(cfg.DEVICE)

    ctrl = build_ctrl(cfg)
    transformer = build_transformer(cfg)
    cuti_module = build_cuti_module(cfg)
    backbone = build_backbone(cfg)
    decoder_layer = cfg.MODELS.TRANSFORMER.DEC_LAYERS

    model = CuTi(backbone, ctrl,transformer, cuti_module,decoder_layer) 

    return model


