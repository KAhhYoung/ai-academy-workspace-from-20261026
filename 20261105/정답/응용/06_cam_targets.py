
"""
Day14 실습: Grad-CAM의 Target 클래스 변경에 따른 시각화 비교

[실습 목적]
동일한 이미지에 대해 서로 다른 클래스의 Logit을 기준으로 Grad-CAM을 생성하여,
분석 대상 클래스에 따라 모델이 강조하는 이미지 영역이 어떻게 달라지는지 확인합니다.

[주요 동작]
1. Train/Valid 이미지 로딩 및 32×32 RGB Tensor 변환
2. TinyShapeCNN 생성 및 사전학습된 Backbone 가중치 로딩
3. Backbone 동결 후 분류 Head만 8 Epoch 학습
4. 동일한 검증 이미지 1장에 대해 Target 클래스 0과 1을 각각 분석
5. 클래스별 Logit의 Gradient를 이용하여 Grad-CAM 생성
6. 원본 이미지와 Heatmap을 합성하여 클래스별 PNG 저장
7. Target별 Logit과 평균 CAM 값을 비교

[주의사항]
- Test 데이터는 사용하지 않습니다.
- 동일한 이미지에서 Target 클래스만 변경하여 비교합니다.
- Target은 반드시 모델이 예측한 클래스일 필요는 없습니다.
- Grad-CAM은 클래스 점수에 대한 양의 기여 영역을 보여주는 기법으로,
  강조된 영역이 실제 정답이나 인과적 판단 근거임을 증명하지는 않습니다.
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
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import cv2

# ============================================================
# 1. 실행 환경 및 경로 설정
# ============================================================

torch.set_num_threads(1)
torch.manual_seed(14)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)


# ============================================================
# 2. 이미지 로딩 및 전처리
# ============================================================

# 교육용 CNN 입력 규격: RGB 32×32, 픽셀값 [0,1]
# ImageNet ResNet18의 Normalize 설정은 사용하지 않음
transform = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor()
])

# Train/Valid 데이터만 로딩 (Test 데이터는 사용하지 않음)
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# 이미지 Tensor와 정답 라벨 구성
X_train = torch.stack([train[i][0] for i in range(len(train))])
y_train = torch.tensor(train.targets, dtype=torch.long)

X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)


# ============================================================
# 3. CNN 모델 생성
# ============================================================

model = nn.Sequential(

    # [Stage 1] 기초 특징 추출
    nn.Conv2d(3, 8, 3, padding=1, bias=False),  # [0]
    nn.BatchNorm2d(8),                          # [1]
    nn.ReLU(),                                  # [2]
    nn.MaxPool2d(2),                            # [3]

    # [Stage 2] 마지막 특징 추출 Stage
    nn.Conv2d(8, 16, 3, padding=1, bias=False), # [4]
    nn.BatchNorm2d(16),                         # [5]
    nn.ReLU(),                                  # [6]

    # [특징 집약] 16×4×4 → 256차원 특징 벡터
    nn.AdaptiveAvgPool2d((4, 4)),               # [7]
    nn.Flatten(),                               # [8]

    # [Head] 3개 클래스의 Logit 출력
    nn.Linear(256, 3)                           # [9]
)


# ============================================================
# 4. 사전학습 Backbone 가중치 로딩
# ============================================================

# 별도 합성 이미지로 사전학습한 가중치를 로딩
# allow_pickle=False: Pickle 객체 역직렬화를 사용하지 않음
with np.load(
    DATA / "source_backbone.npz",
    allow_pickle=False
) as saved:

    state = model.state_dict()

    state.update({
        name: torch.from_numpy(saved[name].copy())
        for name in saved.files
    })

model.load_state_dict(state)


# ============================================================
# 5. Backbone 동결 및 Head 학습 설정
# ============================================================

# 모든 파라미터 동결
for p in model.parameters():
    p.requires_grad = False

# 마지막 Head만 학습 가능하도록 설정
for p in model[9].parameters():
    p.requires_grad = True

# BatchNorm의 Running Mean/Var를 고정
# eval() 상태에서도 Gradient 계산은 가능
model.eval()

# Head 파라미터만 Optimizer에 전달
optimizer = torch.optim.AdamW(
    model[9].parameters(),
    lr=0.01,
    weight_decay=0.0
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. Head Only 학습 (8 Epoch)
# ============================================================

# Backbone은 유지하고 마지막 분류 Head만 학습
for epoch in range(8):

    optimizer.zero_grad()

    # 순전파: 클래스별 Logit 계산 및 손실 계산
    loss = criterion(model(X_train), y_train)

    # 역전파 및 Head 가중치 갱신
    loss.backward()
    optimizer.step()


# ============================================================
# 7. 검증 이미지 예측
# ============================================================

with torch.no_grad():

    # 검증 이미지의 클래스별 Logit 계산
    logits = model(X_valid)

    # Softmax로 클래스별 확률 계산
    probabilities = logits.softmax(dim=1)

    # 가장 높은 Logit을 가진 클래스 선택
    predictions = logits.argmax(dim=1)


# ============================================================
# 8. 동일 이미지에서 Target 클래스별 Grad-CAM 비교
# ============================================================

# 검증 데이터의 첫 번째 이미지를 분석 대상으로 고정
# Target만 변경하여 CAM의 차이를 비교
index = 0

# 클래스 0(circle), 클래스 1(square)을 각각 분석
for target in [0, 1]:

    # --------------------------------------------------------
    # 8-1. Grad-CAM 분석용 입력 준비
    # --------------------------------------------------------

    # 동일한 검증 이미지 1개를 Batch 형태로 추출
    # requires_grad=True: Gradient 계산 경로 활성화
    image = X_valid[index:index + 1].clone().requires_grad_(True)

    # --------------------------------------------------------
    # 8-2. Feature Map 및 클래스별 Logit 계산
    # --------------------------------------------------------

    # 마지막 Conv 블록의 출력 Feature Map 추출
    # Shape: [1, 16, 16, 16]
    features = model[:7](image)

    # AdaptiveAvgPool → Flatten → Head를 거쳐 Logit 계산
    # Shape: [1, 3]
    out = model[7:](features)

    # --------------------------------------------------------
    # 8-3. Target 클래스에 대한 Gradient 계산
    # --------------------------------------------------------

    # 선택한 target 클래스의 Logit을 Feature Map에 대해 미분
    #
    # target=0: circle 클래스 점수의 Gradient
    # target=1: square 클래스 점수의 Gradient
    #
    # 동일 이미지라도 Target 클래스가 달라지면
    # Gradient와 CAM이 달라질 수 있음
    gradient = torch.autograd.grad(
        out[0, target],
        features
    )[0]

    # --------------------------------------------------------
    # 8-4. Grad-CAM 생성
    # --------------------------------------------------------

    # Gradient를 공간 차원(H,W)에 대해 평균하여
    # 채널별 중요도 가중치 계산
    #
    # 가중치와 Feature Map을 곱한 후 채널 방향으로 합산
    # ReLU로 음수 반응을 제거하고 양의 기여만 남김
    cam = (
        gradient.mean((2, 3), keepdim=True) * features
    ).sum(1).relu()

    # --------------------------------------------------------
    # 8-5. CAM 크기 변환 및 정규화
    # --------------------------------------------------------

    # CAM 16×16을 원본 입력 크기 32×32로 확대
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(1),
        (32, 32),
        mode="bilinear",
        align_corners=False
    )[0, 0].detach().numpy()

    # CAM을 [0,1] 범위로 정규화
    # 1e-8은 0으로 나누는 오류 방지
    cam = cam / max(float(cam.max()), 1e-8)

    # --------------------------------------------------------
    # 8-6. 원본 이미지 및 Heatmap 생성
    # --------------------------------------------------------

    # Tensor [C,H,W] → 이미지 [H,W,C]
    # 픽셀값 [0,1] → [0,255]로 변환
    rgb = np.uint8(
        X_valid[index].permute(1, 2, 0).numpy() * 255
    )

    # CAM을 JET ColorMap으로 변환
    # 높은 CAM 값은 상대적으로 따뜻한 색으로 표현
    heat = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # 8-7. 원본 이미지와 Heatmap 합성
    # --------------------------------------------------------

    # 원본 이미지 60% + Heatmap 40% 합성
    # OpenCV는 BGR 채널 순서를 사용하므로 RGB → BGR 변환
    overlay = cv2.addWeighted(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        0.6,
        heat,
        0.4,
        0
    )

    # --------------------------------------------------------
    # 8-8. Grad-CAM 결과 저장
    # --------------------------------------------------------

    # 결과를 256×256으로 확대하여 PNG 저장
    #
    # target_0_cam.png: circle 기준 Grad-CAM
    # target_1_cam.png: square 기준 Grad-CAM
    cv2.imwrite(
        str(OUT / f"target_{target}_cam.png"),
        cv2.resize(overlay, (256, 256))
    )

    # --------------------------------------------------------
    # 8-9. 클래스별 분석 결과 출력
    # --------------------------------------------------------

    # logit: 해당 클래스의 원시 예측 점수
    # mean CAM: 정규화된 CAM의 전체 공간 평균
    #
    # 주의: Logit은 확률이 아니며,
    # mean CAM은 클래스별 신뢰도나 정확도를 의미하지 않음
    print(
        "target:", valid.classes[target],
        "logit:", round(out[0, target].item(), 4),
        "mean CAM:", round(float(cam.mean()), 4)
    )
