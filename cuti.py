from ctrlc import transformer
import torch
import torch.nn as nn
import lietorch
import numpy as np
import numpy.linalg as LA
from lietorch import SE3

from util.position_encoding import build_pos_sine

from typing import Optional
import torch.nn.functional as F
import torch.linalg

# import pytorch3d
# from pytorch3d import transforms

from ctrlc import build_ctrl
from ctrlc.backbone import build_backbone
from ctrlc.transformer import build_transformer
from loftr import build_cuti_module
from loftr import build_cuti_encoder
from einops.einops import rearrange
# import torch_geometric as tgm
from config import cfg


class CuTi(nn.Module):
    def __init__(self, ctrlc1,ctrlc2, cuti_encoder1,cuti_encoder2,cuti_module1,cuti_module2, decoder_layer,PositionEncodingSine):
        super().__init__()
        self.pose_size = 7
        self.num_images = 2
        
        #self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.line_num = 512
        self.hidden_dim = 256

        self.embedding_layer = nn.Embedding(self.num_images,self.hidden_dim)
        self.position_embedding = nn.Embedding(self.line_num,self.hidden_dim)

        self.feature_resolution = (15, 20)
        
        self.intrinsic_matrix = torch.tensor([[517.97,0,320],
                                          [0,517.97,240],
                                            [0,0,1]])
        
        self.ctrlc1 = ctrlc1
        self.ctrlc2 = ctrlc2

        self.cuti_encoder1 = cuti_encoder1
        self.cuti_encoder2 = cuti_encoder2

        self.cuti_module1 = cuti_module1
        self.cuti_module2 = cuti_module2
        self.decoder_layer = decoder_layer

        self.pos_encoding = PositionEncodingSine

        #mlp
        self.mlp1 = nn.Sequential(
            nn.Linear(6,6),
            nn.ReLU(),
            nn.Linear(6,1)

        )
        self.mlp2 = nn.Sequential(
            nn.Linear(6,6),
            nn.ReLU(),
            nn.Linear(6,1)

        )

        #demension check
        self.layer1 = nn.Sequential(
            nn.Linear(2*self.hidden_dim,2*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2*self.hidden_dim,self.hidden_dim)
        )
        #dimension
        self.pool_feat1 = 32
        self.pool_feat2 = 16

        self.K = int(self.num_images*self.pool_feat2*(self.line_num))
        self.K2 = 60
        #pos_regressor
        self.pool_attn1 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.pool_feat1, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.pool_feat1),
            nn.ReLU(),
            nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.pool_feat2)
        )
        self.pose_regressor1 = nn.Sequential(
            nn.Linear(self.K, self.K2),
            nn.ReLU(),
            nn.Linear(self.K2, self.K2),
            nn.ReLU(), 
            nn.Linear(self.K2, self.num_images * 7),
            nn.Unflatten(1, (self.num_images, 7))
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
        
        lines = lines.permute(1,0,2,3) #[img_num,B,line_num,3]
        images = images.permute(1,0,2,3,4) #[img_num,B,C,H,W]
        
        #CTRL_C 모델에 Line,image 정보 넣어주기
        #hs1,memory1,pred1_zvp,pred1_h1vp,pred1_h2vp,pred1_vw,pred1_h1w,pred1_h2w
        hs1,memory1,pred1_vp1,pred1_vp2,pred1_vp3= self.ctrlc1(          #[dec_lay,bs,line_num,hidden_dim]
            images[0],
            lines[0]
            )
        #hs2,memory2,pred2_zvp,pred2_h1vp,pred2_h2vp,pred2_vw,pred2_h1w,pred2_h2w
        hs2,memory2,pred2_vp1,pred2_vp2,pred2_vp3 = self.ctrlc2(          #[dec_lay,bs,line_num,hidden_dim]
            images[1],
            lines[1]
        )

        self.batch_size = hs1.size(1)
        ## ctrl-c에서 dec_layer의 모든 출력이 stack되서 나온 출력을 mlp를 통과시켜서 하나로 만들고 차원 축소해주기
        ## [dec_lay,Batch_size,feature_resolution,hidden_dim] = > [Batch_size,feature_resolution,hidden_dim] 
        hs1 = hs1.permute(1,2,3,0)
        hs2 = hs2.permute(1,2,3,0)
        hs1 = self.mlp1(hs1)
        hs2 = self.mlp2(hs2)
        hs1 = hs1.squeeze(3)
        hs2 = hs2.squeeze(3)

        x = torch.tensor([0],dtype=torch.long).cuda()
        y = torch.tensor([1],dtype=torch.long).cuda()
        p = torch.arange(0,512,1,dtype=torch.long).cuda()

        hs1 = hs1 + self.embedding_layer(x) + self.position_embedding(p)
        hs2 = hs2 + self.embedding_layer(y) + self.position_embedding(p)
        
        line1_embedding = self.embedding_layer(x) + self.position_embedding(p)
        line2_embedding = self.embedding_layer(y) + self.position_embedding(p)    
        # memory1 = memory1 + self.embedding_layer(x)
        # memory2 = memory2 + self.embedding_layer(x)

        #print("memory1.shape",memory1.shape)
        # feature information 차원 맞추기
        # memory1 = memory1.reshape(self.batch_size,-1,self.hidden_dim) 
        # memory2 = memory2.reshape(self.batch_size,-1,self.hidden_dim)

        #image feature, line feature concatenate
        F1 = hs1 #torch.cat([memory1,hs1],dim=1)
        F2 = hs2 #torch.cat([memory2,hs2],dim=1)

        # print("F1.shape",F1.shape)
        # print("F2.shape",F2.shape)
       #self.cuti_encoder1(F1,F2,None,None,line1_embedding,line2_embedding)
        F1,F2 = self.cuti_encoder1(F1,F2,None,None,None,None)

        output = torch.cat([F1.unsqueeze(0),F2.unsqueeze(0)],dim=0)
        output = output.reshape([self.batch_size,self.num_images,self.line_num,-1]).permute([0,3,1,2])

        pooled_output1 = self.pool_attn1(output)
        pose_preds1 = self.pose_regressor1(pooled_output1.reshape([self.batch_size, -1]))

        
        return self.normalize_preds(Gs, pose_preds1)#,pred1_vp1,pred1_vp2,pred1_vp3,pred2_vp1,pred2_vp2,pred2_vp3#,pred1_vw_count,pred1_h1w_count,pred1_h2w_count,pred2_vw_count,pred2_h1w_count,pred2_h2w_count#,R21_to_Q,R22_to_Q,R23_to_Q,R24_to_Q
        



def build(cfg):
    device = torch.device(cfg.DEVICE)

    ctrl1 = build_ctrl(cfg)
    ctrl2 = build_ctrl(cfg)
    checkpoint = torch.load("/home/kmuvcl/source/oldCuTi/CuTi/checkpoint/checkpoint0099.pth")


    ctrl1.load_state_dict(checkpoint['model'],False)
    ctrl2.load_state_dict(checkpoint['model'],False)
    
    ctrl1.eval()
    ctrl2.eval()
    print("CTRL_C model load & Freeze")
    print("only line feature use")
    print("end_to_end")
    # del checkpoint
    cuti_module1 = None#build_cuti_module(cfg)
    cuti_module2 = None#build_cuti_module(cfg)
    cuti_encoder1 = build_cuti_encoder(cfg)
    cuti_encoder2 = None#build_cuti_encoder(cfg)
    PositionEncodingSine = build_pos_sine(cfg)
    decoder_layer = cfg.MODELS.TRANSFORMER.DEC_LAYERS

    model = CuTi(ctrl1,ctrl2,cuti_encoder1,cuti_encoder2, cuti_module1,cuti_module2,decoder_layer,PositionEncodingSine) 

    return model


