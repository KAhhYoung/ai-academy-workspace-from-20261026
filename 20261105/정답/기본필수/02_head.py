"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
"""
Day14 실습: 전이학습(Transfer Learning)의 기본 구조 이해

[실습 목적]
사전학습된 CNN의 특징 추출 계층(Backbone)을 재사용하고,
새로운 분류 계층(Head)만 학습하도록 구성하는 전이학습 원리를 이해합니다.

[주요 동작]
1. TinyShapeCNN 모델을 생성하고 제공된 사전학습 가중치를 불러옵니다.
2. Backbone의 가중치를 동결(Freeze)하여 학습되지 않도록 설정합니다.
3. 기존 출력 계층을 3개 클래스(circle, square, triangle) 분류용 Head로 교체합니다.
4. 학습 가능한 파라미터 수와 모델 출력 Shape을 확인합니다.
5. ResNet18을 생성하여 Backbone 동결 및 Head 교체 구조를 비교합니다.

[주의사항]
- 본 스크립트는 실제 모델 학습이 아닌 전이학습 구조를 준비하고 확인하는 과정입니다.
- TinyShapeCNN은 별도의 합성 이미지로 사전학습된 가중치를 사용합니다.
- ResNet18은 사전학습 가중치 없이 구조만 확인하며 실제 전이학습 성능을 평가하지 않습니다.
- Train/Valid/Test 이미지를 읽거나 모델 학습을 수행하지 않습니다.
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
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
model = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
)
# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    state = model.state_dict()
    state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
model.load_state_dict(state)

for p in model.parameters():
    p.requires_grad = False
# TODO B02: 256개 특징에서 3개 logit을 출력하는 새 Linear Head를 만드세요.
# BEGIN B02
model[9] = nn.Linear(256, 3)
# END B02
print("source 종류: 별도 합성 이미지로 사전학습한 TinyShapeCNN")
print("학습 파라미터:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("출력 shape:", tuple(model.eval()(torch.zeros(2, 3, 32, 32)).shape))
# PPT ResNet18의 구조도 다운로드 없이 확인합니다. 이것은 '전이학습 성능'이 아닙니다.
from torchvision.models import resnet18
resnet = resnet18(weights=None)
for p in resnet.parameters():
    p.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 3)
print("ResNet18 구조 확인용 Head 파라미터:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))
print("주의: 위 ResNet18은 구조만 확인했습니다. ImageNet 학습 결과가 아닙니다.")
