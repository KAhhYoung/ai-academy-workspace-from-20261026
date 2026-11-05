
"""
Day14 실습: 불균형 데이터에서 Accuracy와 Macro F1 비교

[실습 목적]
클래스별 데이터 수가 불균형할 때 Accuracy(정확도)만으로
모델 성능을 평가하면 잘못된 판단을 할 수 있음을 이해합니다.

[주요 동작]
1. 클래스별 데이터 수가 서로 다른 가상의 정답 라벨을 생성합니다.
2. 모든 데이터를 다수 클래스(circle)로 예측한 상황을 가정합니다.
3. Accuracy와 Macro F1을 계산하여 비교합니다.
4. Classification Report로 클래스별 Precision, Recall, F1을 확인합니다.

[주의사항]
- 실제 CNN 모델을 학습하거나 추론하는 코드가 아닙니다.
- 성능평가 지표를 이해하기 위한 가상의 예측 결과입니다.
- 불균형 데이터에서는 Accuracy뿐 아니라 Macro F1도 함께 확인해야 합니다.
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

torch.set_num_threads(1)
torch.manual_seed(14)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE

# 교육용 CNN의 이미지 전처리 설정
# RGB 이미지를 32×32 크기로 변환한 뒤 [0,1] Tensor로 변환
# 이번 실습에서는 실제 이미지 전처리를 수행하지 않음
transform = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor()
])


# ============================================================
# 1. 불균형 데이터의 정답 라벨 생성
# ============================================================

# 총 21개의 가상 정답 데이터
# 0: circle   → 12개 (다수 클래스)
# 1: square   →  6개
# 2: triangle →  3개 (소수 클래스)
y_true = [0] * 12 + [1] * 6 + [2] * 3

# 모델이 모든 데이터를 circle(0)로 예측했다고 가정
# circle 12개는 맞히지만 square와 triangle은 모두 오분류
y_pred = [0] * 21

# 클래스별 실제 데이터 수 확인
print("클래스 수:", [y_true.count(i) for i in range(3)])


# ============================================================
# 2. Accuracy(정확도) 계산
# ============================================================

# Accuracy = 전체 예측 중 정답을 맞힌 비율
#
# 전체 21개 중 circle 12개만 정답
# Accuracy = 12 / 21 ≈ 0.5714 (57.14%)
#
# 주의: square와 triangle을 하나도 맞히지 못해도
# 다수 클래스 덕분에 정확도가 57.14%로 나타남
print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 4)
)


# ============================================================
# 3. Macro F1 계산
# ============================================================

# F1 Score: Precision(정밀도)과 Recall(재현율)의 조화평균
#
# Macro F1:
# 각 클래스의 F1을 개별적으로 계산한 후 단순 평균
# 클래스별 데이터 개수와 관계없이 동일한 비중으로 평가
#
# labels=[0,1,2]: 평가할 클래스 지정
# average="macro": 클래스별 F1의 단순 평균
# zero_division=0: 계산 시 0으로 나누는 상황의 결과를 0으로 처리
#
# 모든 데이터를 circle로 예측했기 때문에
# square와 triangle의 F1은 0이 됨
macro = f1_score(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

print("Macro F1:", round(macro, 4))


# ============================================================
# 4. 클래스별 상세 성능 보고서 출력
# ============================================================

# classification_report():
# 각 클래스별 Precision, Recall, F1, Support를 출력
#
# Precision: 해당 클래스로 예측한 것 중 실제 정답 비율
# Recall   : 실제 해당 클래스 중 올바르게 예측한 비율
# F1       : Precision과 Recall의 조화평균
# Support  : 실제 정답 데이터 개수
#
# digits=3: 소수점 셋째 자리까지 출력
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["circle", "square", "triangle"],
        zero_division=0,
        digits=3
    )
)

"""
20261105, TASK001
결론

본 실습에서 전체 정확도는 57.14%였으나, Macro F1은 24.24%로 나타났다. 
이는 다수 클래스인 circle만 예측하고 square와 triangle은 전혀 구분하지 못했기 때문이다.

따라서 불균형 데이터셋에서는 Accuracy만으로 모델 성능을 판단하기 어렵고, 
Macro F1 및 클래스별 Precision·Recall을 함께 평가해야 한다.
"""

