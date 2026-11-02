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

from PIL import Image, ImageDraw

path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]
image = Image.open(path).convert("RGB")
# 좌우반전은 가로·세로 줄무늬 라벨을 보존한다. 90도 회전은 라벨을 바꾸므로 사용하지 않는다.
train_transform = transforms.Compose([
    transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15), transforms.ToTensor(),
])

# BEGIN A04
# 검증(Validation) 데이터에 적용할 이미지 전처리 파이프라인을 정의합니다.
# transforms.Compose([...]) : 여러 전처리 작업을 순서대로 실행하도록 묶어줍니다.
# transforms.Resize((64, 64)) : 입력 이미지의 높이(H)와 너비(W)를 각각 64픽셀로 조정합니다.
#                                원본 이미지 크기와 관계없이 64×64로 변환합니다.
# transforms.ToTensor()       : PIL 이미지 또는 NumPy 배열을 PyTorch 텐서로 변환합니다.
#                                이미지 차원 순서를 HWC → CHW로 변경합니다.
#                                uint8 이미지의 픽셀값을 0~255에서 0.0~1.0 범위로 정규화합니다.
# 최종 출력 Shape             : RGB 이미지 기준 (3, 64, 64)
# 참고                       : 검증 데이터에는 일반적으로 무작위 증강을 적용하지 않아 일관된 평가 조건을 유지합니다.
valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# END A04
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
