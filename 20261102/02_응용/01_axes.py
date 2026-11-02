from pathlib import Path
import copy
import csv
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.set_num_threads(1)

root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]

# PIL -> Numpy : 축순서(H,W,C) / 색상 순서(RGB)
# PIL -> transforms.ToTensor() : 축순서(C,H,W) / 색상 순서(RGB)
chw = transforms.ToTensor()(Image.open(path).convert("RGB"))

# TODO A01: 채널 축을 마지막으로 보내 HWC를 만든다.
# unsqueeze(dim): 지정한 위치에 크기가 1인 축을 추가하는 메서드 / 새로운 값으로 채워지는 것이 아님
# ex. hwc = chw.permute한 결과: (64,64,3) -> hwc.unsqueeze(0): (1,64,64,3)
hwc = chw.permute(1, 2, 0)
nchw = hwc.permute(2, 0, 1).unsqueeze(0)

plt.figure(figsize=(4, 4))
plt.imshow(hwc.numpy())
plt.title("HWC view: horizontal stripes")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "image_axes.png", dpi=120)
plt.close()

print("CHW / HWC / NCHW:", tuple(chw.shape), tuple(hwc.shape), tuple(nchw.shape))
print("왕복 값 일치:", torch.equal(chw, nchw[0]))
