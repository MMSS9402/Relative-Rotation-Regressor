import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer

class GPTran(nn.Module):
    def __init__(self, backbone, transformer, num_queries, 
                 aux_loss=False, use_structure_tensor=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.use_structure_tensor = use_structure_tensor

        hidden_dim = 256
        self.vp1_embed = nn.Linear(hidden_dim, 3)
        self.vp2_embed = nn.Linear(hidden_dim, 3)
        self.vp3_embed = nn.Linear(hidden_dim, 3)
        self.vp1_class_embed = nn.Linear(hidden_dim, 1)
        self.vp2_class_embed = nn.Linear(hidden_dim, 1)
        self.vp3_class_embed = nn.Linear(hidden_dim, 1)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        #self.line_embed = nn.Embedding(512, hidden_dim)
        line_dim = 3
        if self.use_structure_tensor:
            line_dim = 6        
        self.input_line_proj = nn.Linear(line_dim, hidden_dim)        
        self.backbone = backbone
        self.aux_loss = aux_loss   

    def forward(self, lines, desc):
#     def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels            
        """
        extra_info = {}
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.backbone(samples)

        # src, mask = features[-1].decompose()
        # assert mask is not None
        lmask = None #~extra_samples['line_mask'].squeeze(2).bool()
        desc = desc

        # vlines [bs, n, 3]
        if self.use_structure_tensor:
            lines = self._to_structure_tensor(lines)
        tgt = self.input_line_proj(lines)

        bs = tgt.shape[0]
        
        # desc = rearrange(desc,'b l p c -> (b l) c p').contiguous()
        # desc = F.max_pool1d(desc, kernel_size=21, stride=1)
        # desc = rearrange(desc, '(b l) c p -> b l p c',l=250,b=bs).contiguous().squeeze(2)
        
        tgt = tgt + desc
        
        query_embed=self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = tgt.permute(1, 0, 2)
        query_pos = torch.cat([query_embed, torch.zeros_like(tgt)], dim=0)
        tgt = torch.cat([torch.zeros_like(query_embed), tgt], dim=0)
        
        hs, enc_attn = self.transformer(src=tgt, src_key_padding_mask=None, pos=query_pos)
        # hs, memory, enc_attn, dec_self_attn, dec_cross_attn = (
        #     self.transformer(src=self.input_proj(src), mask=None,
        #                      query_embed=self.query_embed.weight,
        #                      tgt=self.input_line_proj(lines), 
        #                      tgt_key_padding_mask=None,
        #                      pos_embed=pos[-1]))#,line_embed = self.line_embed.weight))
        # ha [n_dec_layer, bs, num_query, ch]

        hs = rearrange(hs,'l b h -> b l h').contiguous().unsqueeze(0)

        extra_info['enc_attns'] = enc_attn
        # extra_info['dec_self_attns'] = dec_self_attn
        # extra_info['dec_cross_attns'] = dec_cross_attn
        outputs_vp1 = self.vp1_embed(hs[:,:,0,:]) # [n_dec_layer, bs, 3]
        outputs_vp1 = F.normalize(outputs_vp1, p=2, dim=-1)

        outputs_vp2 = self.vp2_embed(hs[:,:,1,:]) # [n_dec_layer, bs, 3]
        outputs_vp2 = F.normalize(outputs_vp2, p=2, dim=-1)

        outputs_vp3 = self.vp3_embed(hs[:,:,2,:]) # [n_dec_layer, bs, 3]
        outputs_vp3 = F.normalize(outputs_vp3, p=2, dim=-1)  

        outputs_vp1_class = self.vp1_class_embed(hs[:,:,3:,:])
        outputs_vp2_class = self.vp2_class_embed(hs[:,:,3:,:])
        outputs_vp3_class = self.vp3_class_embed(hs[:,:,3:,:])

        pred_view_vps = torch.cat([outputs_vp1[-1].unsqueeze(1),
                                    outputs_vp2[-1].unsqueeze(1),
                                    outputs_vp3[-1].unsqueeze(1)], dim=1)

        
        ctrlc_output = {
                        "pred_view_vps": pred_view_vps,
                        "pred_view_class1": outputs_vp1_class[-1],
                        "pred_view_class2": outputs_vp2_class[-1],
                        "pred_view_class3": outputs_vp3_class[-1],
                        }
        return hs.squeeze(0),ctrlc_output


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


def build_ctrlc(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    ctrlc = GPTran(
        backbone,
        transformer,
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
    )
    return ctrlc
