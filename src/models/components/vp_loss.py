from typing import Callable
import torch
import torch.nn.functional as F


class VpLoss(Callable):
    def __init__(self):
        super().__init__()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def __call__(self, outputs, targets,indices):

        pred_vp = outputs

        tgt_vp = targets

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        cos_sim = F.cosine_similarity(pred_vp[src_idx], tgt_vp[tgt_idx], dim=-1).abs()    
        losses = (1.0 - cos_sim).mean()
    
        return losses
    
# class line_classification_loss(Callable):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, outputs, targets,indices):


