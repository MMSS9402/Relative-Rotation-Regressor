import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Optional
from . import cuti_attention as mca


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    # Deepcopy로 모듈을 N개만큼 넣어서 ModuleList로 만들어줌
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CuTiDecoderLayer(nn.Module):
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
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.self_attn = mca.CuTiSelfAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = mca.CuTiCrossAttention(d_model, nhead, dropout=dropout)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        # self-attention
        tgt2, self_attn_weight = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        # skip-connection
        tgt = tgt + self.dropout1(tgt2)
        # normalization
        tgt = self.norm1(tgt)
        # cross-attention(key,value가 memory로부터 옴 => encoder layer에서 src가 들어옴)
        tgt2, cross_attn_weight = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # skip-connection
        tgt = tgt + self.dropout2(tgt2)
        # normalization
        tgt = self.norm2(tgt)
        # Feedforward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # skip-conncetion & normalization
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class CuTiTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # Decoder layer를 갯수만큼 Deepcopy로 복제해서 modulelist로 받아옵니다.
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # target에 object query와 Line segment 정보를 받고, memory 에는 encoder의 결과물을 받음
        output = tgt

        intermediate = []
        self_attn_weights = []
        cross_attn_weights = []

        # decoder layer를 반복하면서 이전 layer의 output 출력을 이후 layer에 넣어줌
        for layer in self.layers:
            output, self_attn_weight, cross_attn_weight = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            # intermediate는 layer 중간중간 output들을 다 쌓아서 저장한 list
            # 마지막 output을 출력할 때 FFN를 5개를 써서 출력하기 때문에
            # 이걸 각각 loss를 나눠서 주려면 intermediate가 필요한 것이 아닐까..?

            if self.return_intermediate:
                intermediate.append(self.norm(output))
            self_attn_weights.append(self_attn_weight)
            cross_attn_weights.append(cross_attn_weight)
        self_attn_weights = torch.stack(self_attn_weights)
        cross_attn_weights = torch.stack(cross_attn_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), self_attn_weights, cross_attn_weights

        return output.unsqueeze(0) #, self_attn_weights, cross_attn_weights


def build_cuti_module(cfg):
    d_model = cfg.MODELS.TRANSFORMER.HIDDEN_DIM
    dropout = cfg.MODELS.TRANSFORMER.DROPOUT
    nhead = cfg.MODELS.TRANSFORMER.NHEADS
    dim_feedforward = cfg.MODELS.TRANSFORMER.DIM_FEEDFORWARD
    num_encoder_layers = cfg.MODELS.TRANSFORMER.ENC_LAYERS
    num_decoder_layers = cfg.MODELS.TRANSFORMER.DEC_LAYERS
    normalize_before = cfg.MODELS.TRANSFORMER.PRE_NORM
    return_intermediate_dec = True
    activation = "relu"
    decoder_norm = nn.LayerNorm(d_model)
    decoder_layer = CuTiDecoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    return CuTiTransformer(
        decoder_layer,
        num_decoder_layers,
        decoder_norm,
        return_intermediate=return_intermediate_dec,
    )
