"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
"""
Day14 실습: 사전학습 모델별 입력 전처리(Preprocessing) 비교

[실습 목적]
CNN 모델마다 요구하는 입력 이미지의 크기와 정규화 방식이 다르다는 것을 이해하고,
사전학습 모델에 적합한 전처리를 적용해야 하는 이유를 학습합니다.

[주요 동작]
1. 교육용 TinyShapeCNN의 입력 전처리(32×32, RGB, [0,1])를 정의합니다.
2. ImageNet 사전학습 ResNet18의 공식 전처리 설정을 불러옵니다.
3. 동일한 검증 이미지를 두 가지 전처리 방식으로 각각 변환합니다.
4. 변환된 Tensor의 Shape, Resize/Crop 크기, 정규화 mean/std를 비교합니다.

[주의사항]
- 모델마다 학습 당시 사용한 입력 크기와 정규화 기준을 맞춰야 합니다.
- ResNet18의 공식 전처리는 Resize, CenterCrop, Normalize 등을 포함합니다.
- 사전학습 가중치를 다운로드하거나 모델 학습·성능 평가를 수행하지 않습니다.
- 검증 이미지 1장만 전처리 비교에 사용하며 Test 데이터는 읽지 않습니다.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

torch.set_num_threads(1)
torch.manual_seed(14)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
from torchvision.models import ResNet18_Weights
from PIL import Image
weights = ResNet18_Weights.IMAGENET1K_V1  # Enum·변환 생성은 다운로드하지 않음
# TODO A02: Weight가 제공하는 transforms()를 가져오세요.
# BEGIN A02
official = weights.transforms()
# END A02
image = Image.open(sorted((DATA/"valid"/"circle").glob("*.png"))[0]).convert("RGB")
print("소형 source 입력:",tuple(transform(image).shape),"범위 [0,1]")
print("공식 ResNet 입력:",tuple(official(image).shape))
print("공식 Resize/Crop:", official.resize_size,official.crop_size)
print("공식 mean/std:", official.mean,official.std)
print("작은 source와 ImageNet의 Normalize를 서로 섞지 않습니다.")

"""
20261105, task01
[결론]

교육용 TinyShapeCNN과 ImageNet 사전학습 ResNet18은 입력 이미지 크기와 정규화 기준이 서로 다르다는 것을 확인하였다.

TinyShapeCNN은 32×32 크기의 [0,1] RGB 이미지를 사용하지만, 
ResNet18은 224×224 크기와 ImageNet의 평균(mean)·표준편차(std)를 이용한 정규화가 필요하다.

따라서 사전학습 모델을 활용할 때는 해당 모델의 학습 조건에 맞는 전처리를 적용해야 하며, 
서로 다른 모델의 전처리 방식을 혼용하면 예측 성능이 저하될 수 있다.
"""