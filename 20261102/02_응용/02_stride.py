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

x = torch.rand(4, 3, 32, 32)
rows = []


# stride 달리 적용하여 변화 확인하기
for stride in [1, 2, 3, 4, 5, 6]:
    torch.manual_seed(42)
    
    # TODO A02: 현재 반복의 stride를 Conv2d에 전달한다.
    conv = nn.Conv2d(in_channels=3,
                     out_channels=8,
                     kernel_size=3,
                     stride=stride,
                     padding=1
    )

    # 순전파 (학습 아님)
    y = conv(x)

    # conv가 가진 모든 파라미터 텐서의 원소 수를 합한 값
    params = sum(p.numel() for p in conv.parameters())
    # 학습 가능한 파라미터의 개수만 세려면
    #params = sum(p.numel() for p in conv.parameters() if p.requires_grad)

    # 1 MAC=곱셈 후 누적 1회. bias 덧셈은 제외하며 batch 전체 값이다.
    # y.numel(): 출력 텐서 y에 들어 있는 전체 원소 개수 / 출력 형태: (B, 8, H, W) -> y.numel() = B * 8 * H * W
    # (3 * 3 * 3): (C_in * Kh, Kw)
    # [이해어려움..] mac: 배치 전체의 컨볼루션 연산량: 출력 원소 개수 × 원소 하나당 연산량
    macs = y.numel() * (3 * 3 * 3)

    rows.append(dict(stride=stride, y_shape=str(tuple(y.shape)), params=params, macs=macs))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "stride_comparison.csv", index=False)
