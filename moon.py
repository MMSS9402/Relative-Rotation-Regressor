from data.factory import dataset_factory
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg
from cuti import build

db = dataset_factory(
            ["matterport"],
            datapath="/home/moon/source/CuTi/matterport",
            subepoch=0,
            is_training=True,
            gpu=0,
            streetlearn_interiornet_type=None,
            use_mini_dataset=False,
        )

train_loader = DataLoader(
            db, batch_size=1, num_workers=1, shuffle=False
        )

model = build(cfg)
with tqdm(train_loader, unit="batch") as tepoch:
    for i_batch, item in enumerate(tepoch):
        images, poses, intrinsics,lines = [x.to("cuda") for x in item]
        poses_est = model(images, lines, intrinsics=intrinsics)