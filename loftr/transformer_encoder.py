import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttentionDualSoftmax, FullAttention
from typing import Optional, List
from torch import nn, Tensor
import torch.nn.functional as F
from . import multi_head_attention as mha


def _get_clones(module, N):
    # Deepcopy로 모듈을 N개만큼 넣어서 ModuleList로 만들어줌
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # get_clone 할 때 Deepcopy로 moduleList를 만들어주어서, self.layer에
        # Deepcopy로 이루어진 각기 다른 인코더 갯수만큼 들어가게 됨
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # backbone에서 처리된 feature가 src로 들어오게 됨
        output = src

        
        #print("encoder",src.shape)

        for layer in self.layers:
            # layer에서 나온 output들을 계속 다음 layer에 넣어줌
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        #print("encoder",output.shape)

        if self.norm is not None:
            output = self.norm(output)

        return output
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = mha.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # Normalization을 dropout보다 앞에 할 거냐 뒤에 할 거냐로 forward pre,post가 나뉘는 듯함.
    # 왜 이렇게 하는거지?
    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # 쿼리,키,밸류를 따로따로 받아주는게 아니라 src로 한 번에 받아줌.
        q = k = self.with_pos_embed(src, pos)
        # q,k,v가 준비된 다음, multi-haed attention 진행
        src2, _ = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        # 임배딩 된 값을 dropout, normalization
        #print(src2.type())
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward Network에 넣어줌.
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Residual connection
        src = src + self.dropout2(src2)
        # Normalization
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, _= self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
def build_transformer_encoder(cfg):
    d_model = cfg.MODELS.D_MODEL # 256
    nhead = cfg.MODELS.NHEAD # 8
    layer_name = 6
    encoder_layer = TransformerEncoderLayer(
        d_model, nhead
    )
    return TransformerEncoder(encoder_layer,layer_name)