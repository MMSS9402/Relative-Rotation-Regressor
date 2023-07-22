import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.linalg as LA
import torch.nn.functional as F

from ctrlc.data import transforms as T


class RGBDAugmentor:
    """perform augmentation on RGB-D video"""

    def __init__(self, reshape_size, datapath=None):
        self.reshape_size = reshape_size
        p_gray = 0.1
        self.augcolor = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4 / 3.14
                ),
                transforms.RandomGrayscale(p=p_gray),
                transforms.ToTensor(),
            ]
        )
        # self.linetrans = transforms.Compose(
        #                             [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                # )

    def coordinate_yup(self,segs,org_h):
        H = np.array([0,org_h,0,org_h])
        segs[:,1] = -segs[:,1]
        segs[:,3] = -segs[:,3]
        return(H+segs)

    def color_transform(self, images):
        """color jittering"""
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd * num)
        images = 255 * self.augcolor(images[[2, 1, 0]] / 255.0)
        return (
            images[[2, 1, 0]].reshape(ch, ht, wd, num).permute(3, 0, 1, 2).contiguous()
        )

    def normalize_segs(self, lines, pp, rho):
        pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
        return rho * (lines - pp)

    def sample_segs_np(self, segs, num_sample, use_prob=True):
        num_segs = len(segs)
        sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
        mask = np.zeros([num_sample, 1], dtype=np.float32)
        if num_sample > num_segs:
            sampled_segs[:num_segs] = segs
            mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
        else:
            # sampled_segs = segs[:num_sample]
            # mask[:num_segs] = np.ones([num_sample, 1], dtype=np.float32)
            lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=-1)
            prob = lengths / np.sum(lengths)
            idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
            sampled_segs = segs[idxs]
            mask = np.ones([num_sample, 1], dtype=np.float32)
        return sampled_segs

    def normalize_safe_np(self, v, axis=-1, eps=1e-6):
        de = LA.norm(v, axis=axis, keepdims=True)
        de = np.maximum(de, eps)
        return v / de

    def segs2lines_np(self, segs):
        ones = np.ones(len(segs))
        ones = np.expand_dims(ones, axis=-1)
        p1 = np.concatenate([segs[:, :2], ones], axis=-1)
        p2 = np.concatenate([segs[:, 2:], ones], axis=-1)
        lines = np.cross(p1, p2)
        return self.normalize_safe_np(lines)

    def __call__(self, images, poses, intrinsics, lines, vps):
        images = self.color_transform(images)

        sizey, sizex = self.reshape_size #480,640
        #print("reshape_size",self.reshape_size)
        scalex = sizex / images.shape[-1]
        #print("sizex",sizex) # 640
        #print("xxx:",images.shape[-1]) #  640
        #print("yyy:",images.shape[-2]) # 480
        scaley = sizey / images.shape[-2]
        xidx = np.array([0, 2])
        yidx = np.array([1, 3])
        intrinsics[:, xidx] = scalex * intrinsics[:, xidx]
        intrinsics[:, yidx] = scaley * intrinsics[:, yidx]

        pp = (images.shape[-1] / 2,images.shape[-2] / 2)
        #print("pp:",pp) # 320,240
        rho = 2.0 / np.minimum(images.shape[-2], images.shape[-1])
        #print("sizey",sizey)
        lines[0] = self.coordinate_yup(lines[0],sizey)
        lines[0] = self.normalize_segs(lines[0], pp=pp, rho=rho)
        lines[0] = self.sample_segs_np(lines[0], 512)
        lines[0] = self.segs2lines_np(lines[0])

        lines[1] = self.coordinate_yup(lines[1],sizey)
        lines[1] = self.normalize_segs(lines[1], pp=pp, rho=rho)
        lines[1] = self.sample_segs_np(lines[1], 512)
        lines[1] = self.segs2lines_np(lines[1])

        
        #print(images.shape)
        images = F.interpolate(images, size=self.reshape_size)
        #print("images",images.shape) #2,3,480,640
        lines = np.array(lines)
        vps = np.array(vps)
        #print("augmentation:",lines.shape)
        return images, poses, intrinsics, lines, vps
