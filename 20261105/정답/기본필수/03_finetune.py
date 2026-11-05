
"""
Day14 실습: Head Only vs Partial Fine-tuning 비교

[실습 목적]
사전학습된 CNN의 Backbone을 고정하고 Head만 학습하는 방식과,
마지막 특징 추출 Stage의 일부까지 학습하는 방식을 비교합니다.

[주요 동작]
1. Train/Valid 이미지 로딩 및 전처리
2. 사전학습된 Backbone 가중치 불러오기
3. Head Only와 Partial Fine-tuning 설정
4. 두 전략을 동일한 조건에서 8 Epoch 학습
5. 학습 파라미터 수와 검증 정확도 비교
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

torch.set_num_threads(1)
torch.manual_seed(14)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE

# ============================================================
# 1. 데이터 로딩 및 전처리
# ============================================================

# 이미지를 RGB 32×32 크기로 변환하고 [0,1] Tensor로 변환
transform = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor()
])

# Train/Valid 데이터 로딩 (Test는 사용하지 않음)
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# 이미지와 정답 라벨을 Tensor로 구성
X_train = torch.stack([train[i][0] for i in range(len(train))])
y_train = torch.tensor(train.targets, dtype=torch.long)

X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)


# ============================================================
# 2. 학습 전략별 반복 실험
# ============================================================

results = []

for strategy in ["head", "partial"]:

    # 두 전략의 초기화 조건을 동일하게 설정
    torch.manual_seed(14)

    # --------------------------------------------------------
    # 3. CNN 모델 구성
    # --------------------------------------------------------

    model = nn.Sequential(

        # [Stage 1] 기초 특징 추출
        # 입력: [B, 3, 32, 32]
        nn.Conv2d(3, 8, 3, padding=1, bias=False),  # [0]
        nn.BatchNorm2d(8),                          # [1]
        nn.ReLU(),                                  # [2]
        nn.MaxPool2d(2),                            # [3]

        # [Stage 2] 고차원 특징 추출 (마지막 Stage)
        # 입력: [B, 8, 16, 16]
        nn.Conv2d(8, 16, 3, padding=1, bias=False), # [4]
        nn.BatchNorm2d(16),                         # [5]
        nn.ReLU(),                                  # [6]

        # [특징 집약] 특징맵을 256차원 벡터로 변환
        nn.AdaptiveAvgPool2d((4, 4)),               # [7]
        nn.Flatten(),                               # [8]

        # [Head] 최종 3개 클래스 분류
        nn.Linear(256, 3),                          # [9]
    )

    # --------------------------------------------------------
    # 4. 사전학습된 Backbone 가중치 불러오기
    # --------------------------------------------------------

    # 별도의 합성 이미지로 사전학습된 가중치 적용
    # allow_pickle=False: Pickle 객체 역직렬화를 허용하지 않음
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
        state = model.state_dict()
        state.update({
            name: torch.from_numpy(saved[name].copy())
            for name in saved.files
        })

    model.load_state_dict(state)

    # --------------------------------------------------------
    # 5. 학습할 계층 설정 (Freeze / Unfreeze)
    # --------------------------------------------------------

    # 모든 계층의 파라미터 동결
    # requires_grad=False: 역전파 시 해당 가중치를 학습하지 않음
    for p in model.parameters():
        p.requires_grad = False

    # [Head Only / Partial 공통]
    # Head(9)만 학습 가능하도록 설정
    for p in model[9].parameters():
        p.requires_grad = True

    # [Partial Fine-tuning]
    # 마지막 Stage에 있는 Conv2d(4)도 추가로 학습
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True

    # BatchNorm의 Running Mean/Var 갱신 방지
    # 동시에 Dropout 등이 있다면 평가 모드로 동작함
    model.eval()

    # --------------------------------------------------------
    # 6. Optimizer 설정 및 학습
    # --------------------------------------------------------

    # AdamW: 경사하강법 기반으로 학습 가능한 가중치를 갱신
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0
    )

    # 8 Epoch 반복 학습
    for epoch in range(8):

        optimizer.zero_grad()  # 이전 Gradient 초기화

        # 순전파 및 CrossEntropy 손실 계산
        loss = nn.CrossEntropyLoss()(
            model(X_train),
            y_train
        )

        loss.backward()   # 역전파: Gradient 계산
        optimizer.step()  # 학습 가능한 가중치 갱신

    # --------------------------------------------------------
    # 7. 검증 데이터 평가
    # --------------------------------------------------------

    # 검증에서는 Gradient 계산 불필요
    with torch.no_grad():

        # 가장 높은 Logit의 클래스를 예측
        pred = model(X_valid).argmax(1)

    # 전략별 학습 파라미터 수와 검증 정확도 저장
    results.append([
        strategy,
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        accuracy_score(y_valid, pred)
    ])


# ============================================================
# 8. 최종 비교 결과 출력
# ============================================================

df = pd.DataFrame(
    results,
    columns=["strategy", "trainable", "valid_accuracy"]
)

print(df.round(4).to_string(index=False))
