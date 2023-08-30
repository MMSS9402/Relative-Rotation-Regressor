from typing import Callable
import torch
from torch import linalg as LA


class RelPoseLoss(Callable):
    def __init__(self, weights):
        self.weights_tr = weights.translation
        self.weights_rot = weights.rotation

    def __call__(self, Gt_pose, pred_pose):
        

        A_Gt_pose_tran = Gt_pose[:,0,:3]
        A_Gt_pose_rot = Gt_pose[:,0,3:]
        

        
        B_Gt_pose_tran = Gt_pose[:,1,:3]
        B_Gt_pose_rot = Gt_pose[:,1,3:]
        
        A_pred_pose_tran = pred_pose[:,0,:3]
        A_pred_pose_rot = pred_pose[:,0,3:]
        
        B_pred_pose_tran = pred_pose[:,1,:3]
        B_pred_pose_rot = pred_pose[:,1,3:]
        
        A_loss_tr = LA.norm(A_Gt_pose_tran-A_pred_pose_tran,dim=-1)
        B_loss_tr = LA.norm(B_Gt_pose_tran-B_pred_pose_tran,dim=-1)
        
        A_loss_rot = torch.acos(2*torch.clamp(torch.abs(torch.pow(torch.sum(torch.multiply(A_Gt_pose_rot,A_pred_pose_rot),dim=-1),2)),-1.0,1.0)-1)
        B_loss_rot = torch.acos(2*torch.clamp(torch.abs(torch.pow(torch.sum(torch.multiply(B_Gt_pose_rot,B_pred_pose_rot),dim=-1),2)),-1.0,1.0)-1)
        
        loss_tr = (A_loss_tr + B_loss_tr).mean()
        loss_rot = (A_loss_rot + B_loss_rot).mean()

        loss = self.weights_tr * loss_tr + self.weights_rot * loss_rot
        loss_dict = {
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
        }
        return loss, loss_dict
