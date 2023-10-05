import pytorch_lightning as pl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datamodule.components.megadepth_dataset import MegaDepthDataset  # MegaDepth 데이터셋 모듈을 import합니다.

'''
MegaDepth 데이터에서 데이터를 가공해서 모델에 밀어넣어주는 부분을 짠다고 생각하시면 됩니다.
CuTi 데이터 모듈에서는 pair (image 정보, Line 정보, 두 이미지 사이의 relative pose 정보, 각 이미지의 pose 정보) 가 있으면 됩니다.
Line 정보는 추후에 LSD(Line Segment Detector)를 활용해서 csv 파일로 만든 다음에 불러올거니까 지금은 생각하지 않으셔도 됩니다.
제 코드에서는 data_module 부분을 보시고 짜시면 됩니다.
(src/datamodule에서 matterport_datamodule.py, matterport_dataset.py, base.py, augmentation.py)
어차피 src/datamodule 위치에 새로 작성하시면 됩니다.
'''

class MegaDepthDataModule(pl.LightningDataModule):
    def __init__(self, 
                data_dir, 
                batch_size, 
                num_workers, 
                image_size=(256, 256),
                ):
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage=None):
        # 데이터셋을 불러오고 분할합니다.
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # 다른 전처리 및 데이터 증강(transform)을 추가할 수 있습니다.
        ])
        self.dataset = MegaDepthDataset(self.data_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # 검증 데이터 로더를 정의하는 메서드
        pass

    def test_dataloader(self):
        # 테스트 데이터 로더를 정의하는 메서드
        pass