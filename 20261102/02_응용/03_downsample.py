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

torch.manual_seed(42)
torch.set_num_threads(1)

root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

x = torch.rand(4, 8, 32, 32)

# 다운샘플링(downsampling) 층: 입력 특징 맵의 높이(H)와 너비(W)를 줄이는 층
# nn.MaxPool2d(kernel_size)
operations = {"pool": nn.MaxPool2d(2),
              "conv_stride2": nn.Conv2d(in_channels=8,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1
)}

rows = []
for name, layer in operations.items():
    # TODO A03: 각 downsampling 층을 입력 x에 적용한다.
    y = layer(x)
    rows.append(dict(method=name,
                     shape=str(tuple(y.shape)),
                     params=sum(p.numel() for p in layer.parameters())
    ))
    
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "downsample_comparison.csv", index=False)
