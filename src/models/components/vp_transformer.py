from typing import List, Optional
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention, BilinearAttention


class VPEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 attention='bilinear'):
        super(VPEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = BilinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.proj = nn.Linear(d_model, d_model)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            source1: torch.Tensor,
            x_mask: Optional[torch.Tensor] = None,
            source_mask: Optional[torch.Tensor] = None,
            x_embedding: Optional[torch.Tensor] = None,
            source_embedding: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
            x_embedding (torch.Tensor): [N, L, C]
            source_embedding (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)
        query, key, value1 = x, source1, x
        value2 = source1

        # multi-head attention
        if x_embedding is not None:
            query += x_embedding
        if source_embedding is not None:
            key += source_embedding

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value1 = self.v_proj(value1).view(bs, -1, self.nhead, self.dim)
        value2 = self.v_proj(value2).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value1, value2 ,q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]

        message = message.transpose(-2,-1)
        message = self.proj(message)

        return message


class VPTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            layer_types: List[str],
            attention: str = "linear",
    ):
        super(VPTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_types = layer_types
        encoder_layers = [
            VPEncoderLayer(d_model, nhead, attention=attention) for _ in self.layer_types
        ]
        self.layers = nn.ModuleList(encoder_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            feat0: torch.Tensor,
            feat1: torch.Tensor,
            mask0: torch.Tensor = None,
            mask1: torch.Tensor = None,
            embedding0: torch.Tensor = None,
            embedding1: torch.Tensor = None,
    ):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
            embedding0 (torch.Tensor): [N, L, C] (optional)
            embedding1 (torch.Tensor): [N, S, C] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'vpcross':
                bi_feat0 = layer(feat0, feat1, mask0, mask1, embedding0, embedding1)
                bi_feat1 = layer(feat1, feat0, mask1, mask0, embedding1, embedding0)
            else:
                raise KeyError

        return bi_feat0, bi_feat1