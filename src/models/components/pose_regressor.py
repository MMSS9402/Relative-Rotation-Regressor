import torch
import torch.nn as nn
from einops import rearrange
from .regressor_layers import DeepResBlock, MLP

class Features2PoseError(nn.Module):
    '''
        Features2PoseError computes the relative pose error for a query fundamental matrix
        given the dense feature extracted from image A and image B. It outputs the translation
        and rotation pose errors independently.
        Arguments:
            conf: Configuration dictionary containing the parameters for the error regressor block.
            d_model: Feature dimension after feature extractor and attention layers (default: 128d).
    '''

    def __init__(self):
        super(Features2PoseError, self).__init__()

        dim_vector = 256
        # self.process_volume = DeepResBlock(d_model, dim_vector, conf['BATCH_NORM'])
        self.flat_conv1 = MLP(dim_vector, 7, False)
        self.flat_conv2 = MLP(dim_vector, 7, False)
        # self.flat_conv_r = MLP(dim_vector, 1, False)
        self.maxpool = torch.nn.MaxPool1d(1030)
        self.linear = nn.Linear(1030,2)

    def forward(self, xa, xb):
        """ Proccess features and compute the order-invariant translation and rotation pose error.
        Args:   
            xa (torch.Tensor): features from image A after common and epipolar attention blocks.
            xb (torch.Tensor): features from image B after common and epipolar attention blocks.
        Outputs:
            x: Translation and rotation errors (N, 2).
        """
        b = xa.size(0)
        x = self.pred_symm_fmat_err(xa, xb)

        return x

    def pred_symm_fmat_err(self, xa, xb):
        """ Predict the order-invariant translation and rotation pose error.
        Args:
            xa (torch.Tensor): Post-processed features from image A.
            xb (torch.Tensor): Post-processed features from image B.
        Outputs:
            x: Translation and rotation errors (N, 2).
        """
        
        # x = torch.cat([xa.unsqueeze(2),xb.unsqueeze(2)],dim=2)
        # # x = x.mean(dim=3, keepdim=True)
        # x = self.flat_conv(x)

        xa = self.flat_conv1(xa) #[b,7,line_num+3]
        xb = self.flat_conv2(xb) #[b,7,line_num+3]
        x = torch.cat([xa,xb],dim=2)
        # x = rearrange(x, "b c l -> b l c").contiguous()
        x = self.maxpool(x)

        return x