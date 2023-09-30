from typing import List, Optional
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            source: torch.Tensor,
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
        query, key, value = x, source, source

        # multi-head attention
        if x_embedding is not None:
            query += x_embedding
        if source_embedding is not None:
            key += source_embedding

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            layer_types: List[str],
            attention: str = "linear",
    ):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_types = layer_types
        encoder_layers = [
            LoFTREncoderLayer(d_model, nhead, attention=attention) for _ in self.layer_types
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
            if layer_type == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0, embedding0, embedding0)
                feat1 = layer(feat1, feat1, mask1, mask1, embedding1, embedding1)
            elif layer_type == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1, embedding0, embedding1)
                feat1 = layer(feat1, feat0, mask1, mask0, embedding1, embedding0)
            else:
                raise KeyError

        return feat0, feat1
