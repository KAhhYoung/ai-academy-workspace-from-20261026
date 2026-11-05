
"""
Day14 실습: 클래스 불균형 대응을 위한 Class Weight 적용 및 성능 비교

[실습 목적]
클래스별 학습 데이터 수가 불균형할 때, 손실함수에 클래스 가중치를 적용하면
모델의 학습과 소수 클래스 예측 성능이 어떻게 달라지는지 확인합니다.

[주요 동작]
1. Train/Valid 이미지 로딩 및 32×32 RGB Tensor 변환
2. 사전학습된 TinyShapeCNN Backbone의 가중치를 불러오고 동결
3. 동일한 Head Only 모델에서 일반 학습과 Class Weight 적용 학습 비교
4. 클래스별 데이터 수의 역수로 가중치를 계산하고 CrossEntropyLoss에 적용
5. 8 Epoch 동안 학습하고 최저 Valid Loss 모델 선택
6. Accuracy, Macro F1, Triangle Recall, 학습 시간 비교

[주의사항]
- 이번 실습에서는 Head만 학습하며 Backbone은 동결합니다.
- Class Weight는 학습 손실함수에만 적용하고 검증 손실에는 적용하지 않습니다.
- Test 데이터는 사용하지 않으며, Valid 데이터로 모델을 평가합니다.
- 클래스 가중치를 적용한다고 반드시 모든 평가 지표가 개선되는 것은 아닙니다.
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
import time

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
# 2. 데이터 로딩 및 전처리
# ============================================================

# 이미지를 RGB 32×32로 변환하고 픽셀값을 [0,1]로 정규화
# ImageNet ResNet18의 전처리 방식과 혼용하지 않음
transform = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor()
])

# 학습 및 검증 데이터 로딩 (Test 데이터는 사용하지 않음)
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# 이미지 Tensor와 정답 라벨 분리
X_train = torch.stack([train[i][0] for i in range(len(train))])
y_train = torch.tensor(train.targets, dtype=torch.long)

X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)


# ============================================================
# 3. 비교할 실험 조건 설정
# ============================================================

# 동일한 Head Only 모델을 두 가지 방식으로 학습
#
# none   : 클래스 가중치를 적용하지 않는 일반 학습
# weight : 클래스별 데이터 수에 반비례하는 가중치 적용
experiments = [
    ("head", "none"),
    ("head", "weight")
]

# 모델별 최종 결과와 Epoch별 검증 손실 저장
rows = []
all_history = []


# ============================================================
# 4. 실험 조건별 모델 생성 및 학습
# ============================================================

for strategy, balance in experiments:

    # 두 실험의 초기 가중치 조건을 동일하게 유지
    torch.manual_seed(14)

    # --------------------------------------------------------
    # 4-1. CNN 모델 생성
    # --------------------------------------------------------

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

        # [특징 집약] 16×4×4 → 256차원 벡터
        nn.AdaptiveAvgPool2d((4, 4)),               # [7]
        nn.Flatten(),                               # [8]

        # [Head] 3개 클래스 분류
        nn.Linear(256, 3),                          # [9]
    )

    # --------------------------------------------------------
    # 4-2. 사전학습 Backbone 가중치 불러오기
    # --------------------------------------------------------

    # 별도 합성 이미지로 사전학습된 가중치 적용
    # allow_pickle=False: Pickle 객체 역직렬화 방지
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
        state = model.state_dict()
        state.update({
            name: torch.from_numpy(saved[name].copy())
            for name in saved.files
        })

    model.load_state_dict(state)

    # --------------------------------------------------------
    # 4-3. 학습할 파라미터 설정 (Freeze / Unfreeze)
    # --------------------------------------------------------

    # 전체 파라미터 동결
    # requires_grad=False: 해당 가중치는 학습하지 않음
    for p in model.parameters():
        p.requires_grad = False

    # Head(9)만 학습 가능하도록 설정
    for p in model[9].parameters():
        p.requires_grad = True

    # Partial Fine-tuning을 사용할 경우 Stage 2 Conv도 학습
    # 이번 실험의 strategy는 "head"만 있으므로 실행되지 않음
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True

    # BatchNorm의 Running Mean/Var 고정
    # 학습 대상인 Head의 Gradient 계산은 정상적으로 수행됨
    model.eval()

    # 학습 가능한 파라미터만 Optimizer에 전달
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.003,
        weight_decay=0.0001
    )

    # --------------------------------------------------------
    # 4-4. 클래스 가중치(Class Weight) 계산
    # --------------------------------------------------------

    # 클래스별 학습 데이터 개수 계산
    # 예: circle=120, square=40, triangle=20
    counts = torch.bincount(
        y_train,
        minlength=3
    ).float()

    # 클래스별 데이터 수의 역수 계산
    # 데이터가 적은 클래스일수록 더 큰 가중치를 부여
    inverse = 1.0 / counts

    # 가중치 평균이 1이 되도록 정규화
    # 클래스 간 상대적인 중요도는 유지
    class_weights = inverse / inverse.mean()

    # --------------------------------------------------------
    # 4-5. 손실함수(CrossEntropyLoss) 설정
    # --------------------------------------------------------

    # balance="none":
    #   모든 클래스에 동일한 가중치 적용
    #
    # balance="weight":
    #   소수 클래스의 오분류에 상대적으로 큰 손실 부여
    #
    # 이를 통해 모델이 다수 클래스에만 편향되는 현상을
    # 완화할 수 있는지 실험
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if balance == "weight" else None
    )

    # --------------------------------------------------------
    # 4-6. 모델 학습 및 최적 Epoch 선택
    # --------------------------------------------------------

    best_loss = float("inf")

    # 학습 소요 시간 측정 시작
    started = time.perf_counter()

    for epoch in range(1, 9):

        # 이전 Epoch의 Gradient 초기화
        optimizer.zero_grad()

        # 순전파: 예측 Logit 계산 후 손실 산출
        loss = criterion(
            model(X_train),
            y_train
        )

        # 역전파 및 가중치 갱신
        loss.backward()
        optimizer.step()

        # 검증 데이터에서는 Gradient를 계산하지 않음
        with torch.no_grad():

            # 클래스 가중치를 적용하지 않은 Valid Loss 계산
            vl = nn.CrossEntropyLoss()(
                model(X_valid),
                y_valid
            ).item()

        # Epoch별 검증 손실 기록
        all_history.append({
            "strategy": strategy,
            "balance": balance,
            "epoch": epoch,
            "valid_loss": vl
        })

        # 현재 Epoch의 Valid Loss가 가장 낮으면
        # 해당 모델 가중치를 복사하여 보관
        if vl < best_loss:
            best_loss = vl
            best_epoch = epoch

            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

    # 전체 학습 시간 측정
    elapsed = time.perf_counter() - started

    # --------------------------------------------------------
    # 4-7. Best Model 복원 및 검증
    # --------------------------------------------------------

    # 최저 Valid Loss를 기록한 Epoch의 가중치 복원
    model.load_state_dict(best_state)

    with torch.no_grad():

        # 클래스별 Logit을 Softmax 확률로 변환
        prob = model(X_valid).softmax(1)

        # 확률이 가장 높은 클래스 선택
        pred = prob.argmax(1)

    # 클래스별 Precision / Recall / F1 계산
    report = classification_report(
        y_valid,
        pred,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # 4-8. 최종 모델 성능 기록
    # --------------------------------------------------------

    rows.append({

        "strategy": strategy,
        "balance": balance,

        # 실제 학습 가능한 파라미터 개수
        "params": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),

        # 최저 Valid Loss를 기록한 Epoch
        "best_epoch": best_epoch,

        # 전체 검증 데이터의 분류 정확도
        "accuracy": accuracy_score(y_valid, pred),

        # 각 클래스의 F1을 동일 비중으로 평균
        "macro_f1": f1_score(
            y_valid,
            pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),

        # 소수 클래스 Triangle의 재현율
        # 실제 Triangle 중 올바르게 예측한 비율
        "triangle_recall": report["2"]["recall"],

        # 학습 소요 시간(초)
        "seconds": elapsed
    })


# ============================================================
# 5. 최종 비교 결과 저장 및 출력
# ============================================================

print("train class_weights:", class_weights.tolist())

# 실험 결과를 DataFrame으로 변환
table = pd.DataFrame(rows)

# 전체 결과를 CSV로 저장
table.to_csv(
    OUT / "weight_comparison.csv",
    index=False
)

# 학습 시간(seconds)을 제외한 성능 비교표 출력
print(
    table.drop(columns="seconds")
    .round(4)
    .to_string(index=False)
)


"""
20261105,   task001
최종 결론: 

이번 실험에서는 클래스 가중치 적용이 불균형 데이터의 분류 성능 개선에 효과적이었다. 
다만 검증 데이터에서 정확도 100%를 달성했다고 해서 일반화 성능이 보장되는 것은 아니므로, 
향후 독립적인 Test 데이터와 반복 실험을 통해 성능을 추가 검증해야 한다.
"""