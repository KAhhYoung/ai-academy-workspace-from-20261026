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
from PIL import Image, ImageDraw

torch.manual_seed(42)
torch.set_num_threads(1)

root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]
image = Image.open(path).convert("RGB")
# 좌우반전은 가로·세로 줄무늬 라벨을 보존한다. 90도 회전은 라벨을 바꾸므로 사용하지 않는다.
# Q. 가로·세로 줄무늬 라벨을 좌우반전하는 목적은? 
# 줄무늬의 방향은 유지하면서, 위치나 세부 모양이 달라져도 같은 종류로 인식하도록 학습시키려는 목적
# 실제 이미지에는 줄무늬 외에도 다음 요소가 있을 수 있습니다.
# - 줄무늬의 위치와 간격 차이
# - 좌우 밝기 차이
# - 노이즈나 작은 무늬
# 좌우 반전하면 이런 요소의 위치가 달라져서, 모델이 특정 위치나 우연한 패턴에 의존하는 것을 줄이는 데 도움이 될 수 있습니다.

# transforms.ColorJitter(brightness=0.15): 밝기 배율을 0.85(1-0.15)~1.15(1+0.15) 사이에서 무작위로 선택한다는 뜻
# transforms.ToTensor(): 정규화 포함 (0~255 -> 0~1 로 스케일링)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15), transforms.ToTensor(),
])

# TODO A04: valid에는 Resize와 ToTensor만 적용한다.
# 보통 train과 valid 이미지는 같은 크기로 resize합니다. 같은 모델에 같은 입력 크기 조건으로 넣어 평가하기 위해서.
valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

augmented = [train_transform(image) for _ in range(3)]
v1, v2 = valid_transform(image), valid_transform(image)
canvas = Image.new("RGB", (4*128, 150), "white")
draw = ImageDraw.Draw(canvas)
for i, tensor in enumerate([v1]+augmented):
    canvas.paste(transforms.ToPILImage()(tensor).resize((128,128)), (128*i,22))
    draw.text((128*i+5,4), "valid" if i==0 else f"train {i}", fill="black")
canvas.save(out / "augmentation_preview.png")
print("valid 동일:", torch.equal(v1, v2))
print("shape:", tuple(v1.shape))
print("train 3개 동일 여부:", torch.equal(augmented[0], augmented[1]))
