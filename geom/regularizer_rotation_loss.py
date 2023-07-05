# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from config import cfg

class Regularizer(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_vp: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_vp

        self.thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
        
    def loss_vp(self, outputs, targets, **kwargs):
        src_zvp = outputs                                   
        target_zvp = targets

        cos_sim = F.cosine_similarity(src_zvp, target_zvp, dim=-1).abs()     
        #print(cos_sim) 
        loss_vp_cos = (1.0 - cos_sim)
            
        return loss_vp_cos
    
    def relative_l2_loss(self, outputs, targets, **kwargs):
            
        return 
    

    @torch.no_grad()
    def forward(self, pred1_vp1, pred1_vp2, pred1_vp3, pred2_vp1, pred2_vp2, pred2_vp3, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        #1번 경우의 수
        R11 = torch.cat([pred1_vp1,-pred1_vp2,pred1_vp3])
        R12 = torch.cat([-pred1_vp1,-pred1_vp2,pred1_vp3])
        R13 = torch.cat([pred1_vp1,-pred1_vp2,-pred1_vp3])
        R14 = torch.cat([-pred1_vp1,-pred1_vp2,-pred1_vp3])

        #2번 경우의 수
        R21 = torch.cat([pred2_vp1,-pred2_vp2,pred2_vp3])
        R22 = torch.cat([-pred2_vp1,-pred2_vp2,pred2_vp3])
        R23 = torch.cat([pred2_vp1,-pred2_vp2,-pred2_vp3])
        R24 = torch.cat([-pred2_vp1,-pred2_vp2,-pred2_vp3])

        RR1 = R21.T @ R11
        RR2 = R21.T @ R12
        RR3 = R21.T @ R13
        RR4 = R21.T @ R14
        RR5 = R22.T @ R11
        RR6 = R22.T @ R12
        RR7 = R22.T @ R13
        RR8 = R22.T @ R14
        RR9 = R23.T @ R11
        RR10 = R23.T @ R12
        RR11 = R23.T @ R13
        RR12 = R23.T @ R14
        RR13 = R24.T @ R11
        RR14 = R24.T @ R12
        RR15 = R24.T @ R13
        RR16 = R24.T @ R14
        
        #print("pred_vp:",pred_vp.shape)
        # Also concat the target labels and boxes
        GT_RR = targets
        bs, num_queries = 6,12
        
        #print("pred_vp.shape:",pred_vp.shape)
        #print("tgt_vp.shape",tgt_vp.shape)

        # print("pred_vp",pred_vp)
        # print("pred_vp.shape",pred_vp.shape)
        # print("tgt_vp",tgt_vp)
        # print("tgt_vp.shape",tgt_vp.shape)
        # Compute the L1 cost between boxes
        loss = nn.MSELoss()
        cost_vp1 = loss(RR1,GT_RR)
        cost_vp2 = loss(RR2,GT_RR)
        cost_vp3 = loss(RR3,GT_RR)
        cost_vp4 = loss(RR4,GT_RR)
        cost_vp5 = loss(RR5,GT_RR)
        cost_vp6 = loss(RR6,GT_RR)
        cost_vp7 = loss(RR7,GT_RR)

        # Final cost matrix
        C = torch.cat([torch.cat([cost_vp1.unsqueeze(1),cost_vp2.unsqueeze(1)],dim=1),cost_vp3.unsqueeze(1)],dim=1)
        #print(C)
        #print(C.shape)
        C = C.view(bs, num_queries, -1).cpu()
        
        #print(bs,num_queries)

        sizes = 6
        #print("sizes",sizes)
        indices = [linear_sum_assignment(c) for i, c in enumerate(C.split(3, -1)[0])]
        #print(indices)
        # for i,c in enumerate(C.split(3,-1)[0]):
        #     print(i)
        #     print(i,c)
        #     print(c.shape)

        # print("indices",indices)
        # print(indices[0])
        #print([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_vp=1)