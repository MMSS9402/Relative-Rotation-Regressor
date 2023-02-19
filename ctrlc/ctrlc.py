import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone
from .transformer import build_transformer


class GPTran(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        use_structure_tensor=True,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # self.num_queries = num_queries
        self.transformer = transformer

        self.use_structure_tensor = use_structure_tensor

        hidden_dim = transformer.d_model

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # query embedding은 nn모듈에서 가져다가 쓴다...
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        line_dim = 3
        if self.use_structure_tensor:
            line_dim = 6
        self.input_line_proj = nn.Linear(line_dim, hidden_dim)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, extra_samples):
        #     def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        extra_info = {}
        # import pdb; pdb.set_trace()
        print("image_shaep:",samples.shape)
        if isinstance(samples, (list, torch.Tensor)):
            # samples는 배치 단위 이미지
            samples = nested_tensor_from_tensor_list(samples)
        # 이미지를 backbone을 통과시켜 feature 뽑고
        
        features, pos = self.backbone(samples)

        # feature 펴서 src 만들기
        src, mask = features[-1].decompose()
        assert mask is not None
        
        print("ctrlc.shape:",extra_samples.shape)
        extra_samples = torch.tensor(extra_samples, dtype = torch.float32)
        lines = extra_samples
        
        lmask = ~extra_samples.squeeze(2).bool()

        # vlines [bs, n, 3]
        if self.use_structure_tensor:
            lines = self._to_structure_tensor(lines)

        # src를 projection 시켜서 transformer에 넣어주기
        # 여기서 query embedding에 들어가고, 이게 transformer decoder에 들어갈 때 tgt로 변수명이 표시됩니다.
        hs, memory, enc_attn, dec_self_attn, dec_cross_attn = self.transformer(
            src=self.input_proj(src),
            mask=mask,
            tgt=self.input_line_proj(lines),
            tgt_key_padding_mask=None,
            pos_embed=pos[-1],
        )
        # ha [n_dec_layer, bs, num_query, ch]

        # extra_info["enc_attns"] = enc_attn
        # extra_info["dec_self_attns"] = dec_self_attn
        # extra_info["dec_cross_attns"] = dec_cross_attn

        return hs 

    def _to_structure_tensor(self, params):
        (a, b, c) = torch.unbind(params, dim=-1)
        return torch.stack([a * a, a * b, b * b, b * c, c * c, c * a], dim=-1)

    def _evaluate_whls_zvp(self, weights, vlines):
        vlines = F.normalize(vlines, p=2, dim=-1)
        u, s, v = torch.svd(weights * vlines)
        return v[:, :, :, -1]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_ctrl(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    ctrl = GPTran(
        backbone,
        transformer,
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
    )
    weight_dict = dict(cfg.LOSS.WEIGHTS)

    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODELS.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = cfg.LOSS.LOSSES
    # criterion = SetCriterion(
    #     weight_dict=weight_dict,
    #     losses=losses,
    #     line_pos_angle=cfg.LOSS.LINE_POS_ANGLE,
    #     line_neg_angle=cfg.LOSS.LINE_NEG_ANGLE,
    # )
    # criterion.to(device)

    return ctrl  # , criterion
