
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

# 난수 고정 및 CPU 연산 스레드 수 제한
torch.manual_seed(42)
torch.set_num_threads(1)

# 현재 파일 경로를 기준으로 결과 저장 폴더 생성
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

# 이미지 전처리: 32×32 크기로 변경 후 텐서로 변환
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 학습 및 검증 데이터 로드 (하위 폴더명이 클래스 레이블)
train = datasets.ImageFolder(root / "data" / "train", transform=transform)
valid = datasets.ImageFolder(root / "data" / "valid", transform=transform)

# DataLoader: 배치 크기 16, 학습 데이터만 무작위 섞기
# generator: 셔플 순서의 재현성을 위한 난수 시드 지정
train_loader = DataLoader(
    train, batch_size=16, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(42)
)
valid_loader = DataLoader(
    valid, batch_size=16, shuffle=False, num_workers=0
)

# CNN 모델 구성
# Conv2d: 이미지 특징 추출 (입력 채널 → 출력 채널)
# ReLU: 음수 값을 0으로 변환하는 활성화 함수
# MaxPool2d(2): 높이와 너비를 각각 절반으로 축소
# AdaptiveAvgPool2d(1): 각 채널의 공간 정보를 평균값 하나로 압축
# Flatten: 다차원 특징을 1차원 벡터로 변환
# Linear(32, 2): 32개 특징을 이용해 2개 클래스의 점수 출력
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
)

# 손실 함수: 다중 클래스 분류용 교차 엔트로피
loss_fn = nn.CrossEntropyLoss()

# 최적화 알고리즘 AdamW: 학습률 0.01, 가중치 감쇠 없음
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.01, weight_decay=0.0
)

# 학습 기록 및 최적 모델 추적 변수
history = []
best_loss = float("inf")

# 총 5번의 Epoch 동안 학습
for epoch in range(1, 6):
    model.train()  # 학습 모드 활성화

    for images, labels in train_loader:
        optimizer.zero_grad()  # 이전 배치의 기울기 초기화

        # 순전파: 예측값 계산 후 정답과 비교하여 손실 계산
        loss = loss_fn(model(images), labels)

        # BEGIN A05
        loss.backward()   # 역전파: 손실에 대한 기울기 계산
        optimizer.step()  # 계산된 기울기로 모델 가중치 갱신
        # END A05

    # Epoch 종료 후 동일한 평가 모드에서 학습/검증 성능 측정
    model.eval()
    row = {"epoch": epoch}

    # 평가 시 기울기 계산 비활성화
    with torch.no_grad():

        # train과 valid 데이터 각각의 손실 및 정확도 계산
        for phase, loader in [
            ("train", DataLoader(train, batch_size=16)),
            ("valid", valid_loader)
        ]:
            total_loss, correct, count = 0.0, 0, 0

            for images, labels in loader:
                logits = model(images)  # 모델 예측 점수 계산

                # 배치 평균 손실 × 배치 샘플 수
                total_loss += loss_fn(logits, labels).item() * len(labels)

                # 가장 높은 점수의 클래스를 선택하여 정답과 비교
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)

            # 전체 샘플 기준 평균 손실과 정확도 저장
            row[phase+"_loss"] = total_loss/count
            row[phase+"_accuracy"] = correct/count

    history.append(row)  # 현재 Epoch의 성능 기록

    # 검증 손실이 이전 최솟값보다 작으면 최적 모델 저장
    if row["valid_loss"] < best_loss:
        best_loss, best_epoch = row["valid_loss"], epoch
        best_state = copy.deepcopy(model.state_dict())

# 검증 손실이 가장 낮았던 모델의 가중치 복원
model.load_state_dict(best_state)
model.eval()

# 최적 모델로 검증 데이터의 정답 및 예측 결과 수집
actual, predicted = [], []

with torch.no_grad():
    for images, labels in valid_loader:
        actual.extend(labels.tolist())
        predicted.extend(model(images).argmax(1).tolist())

# 2×2 혼동행렬 생성 (행: 실제 클래스, 열: 예측 클래스)
cm = torch.zeros(2, 2, dtype=torch.long)

for truth, guess in zip(actual, predicted):
    cm[truth, guess] += 1

# 학습 기록, 최적 Epoch, 혼동행렬 및 클래스별 정답 수 출력
print(pd.DataFrame(history).round(4).to_string(index=False))
print("best epoch:", best_epoch)
print("혼동행렬 행=실제, 열=예측:\n", cm.tolist())
print("클래스별 정답 수:", cm.diag().tolist(), "/", cm.sum(1).tolist())

# Epoch별 학습/검증 손실 및 정확도 저장
pd.DataFrame(history).to_csv(out / "cnn_history.csv", index=False)

# 혼동행렬을 CSV 파일로 저장
pd.DataFrame(
    cm.numpy(), index=train.classes, columns=train.classes
).to_csv(out / "confusion_matrix.csv")

# 검증 이미지별 실제 레이블과 예측 레이블 저장
pd.DataFrame(
    dict(
        file=[Path(p).name for p, _ in valid.samples],
        actual=actual,
        predicted=predicted
    )
).to_csv(out / "cnn_predictions.csv", index=False)
