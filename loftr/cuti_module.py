import copy
from config import cfg
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention, FulllSelfAttention
from .cuti_attention import CuTiSelfAttention,CuTiCrossAttention


class CuTiEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(CuTiEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FulllSelfAttention()#LinearAttention() #if attention == 'linear' else FullAttention()
        self.cross_attention = FullAttention()
        # self.attention = CuTiSelfAttention(d_model,nhead)
        # self.cross_attention = CuTiCrossAttention(d_model,nhead)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, mode=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        if(mode == 'self'):
            bs = x.size(0)
            query, key, value = x, source, source

            # multi-head attention
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
            #message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
            #message = self.attention(query,key,value=value,attn_mask=None,key_padding_mask=None)
            message = self.attention(query,key,value,None,None)
            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            message = self.norm1(message)

            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=2))
            message = self.norm2(message)

            return x + message
        elif(mode == 'cross'):
            bs = x.size(0)
            query, key, value = x, source, source
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
            message = self.cross_attention(query, key, value,None,None)
            #message = self.cross_attention(query,key,value)
            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            message = self.norm1(message)
            return x + message

class CuTiTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, cfg, d_model,nhead,layer_names,encoder_layer):
        super(CuTiTransformer, self).__init__()

        self.cfg = cfg
        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = CuTiEncoderLayer(d_model, nhead, attention = 'linear')
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        #print("cuti_cuti:",feat0.shape)
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0,mode ='self')
                feat1 = layer(feat1, feat1, mask1, mask1,mode ='self')
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1,mode ='cross')
                feat1 = layer(feat1, feat0, mask1, mask0,mode ='cross')
            else:
                raise KeyError

        return feat0, feat1

def build_cuti_module(cfg):
    d_model = cfg.MODELS.D_MODEL
    nhead = cfg.MODELS.NHEAD
    layer_name = cfg.MODELS.LAYER_NAMES
    encoder_layer = CuTiEncoderLayer(
        d_model, nhead, attention = cfg.MODELS.ATTENTION
    )
    return CuTiTransformer(
    cfg,
    d_model,
    nhead,
    layer_name,
    encoder_layer,
    
    )
    
