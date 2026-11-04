
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

# 현재 실습 경로 및 결과 저장 폴더 생성
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

# torchvision의 ResNet18 모델과 사전학습 가중치 불러오기
from torchvision.models import resnet18, ResNet18_Weights

# ImageNet으로 사전학습된 ResNet18 가중치 지정
weights = ResNet18_Weights.IMAGENET1K_V1

# 사전학습 모델에 맞는 이미지 전처리 적용
# 크기 조정, 중앙 자르기(224×224), 텐서 변환, 정규화 수행
train = datasets.ImageFolder(
    root / "data" / "train",
    transform=weights.transforms()
)
valid = datasets.ImageFolder(
    root / "data" / "valid",
    transform=weights.transforms()
)

# 두 가지 전이학습 전략 비교
# head    : 최종 분류층(fc)만 학습
# partial : 최종 분류층(fc)과 마지막 블록(layer4)을 함께 학습
# 1 Epoch는 학습 과정 확인용이며 일반화 성능 비교에는 한계가 있음
rows = []

for mode in ["head", "partial"]:

    # 모델별 실험 재현을 위한 난수 고정
    torch.manual_seed(42)

    # ImageNet 사전학습 ResNet18 모델 생성
    # 가중치가 없으면 다운로드하며 무작위 가중치로 대체하지 않음
    model = resnet18(weights=weights)

    # 모든 계층의 가중치 동결 (학습 대상에서 제외)
    for param in model.parameters():
        param.requires_grad = False

    # 기존 1000개 클래스 분류층을 2개 클래스 분류층으로 교체
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 최종 분류층(fc)의 학습률을 0.001로 설정
    groups = [
        {"params": model.fc.parameters(), "lr": 0.001}
    ]

    if mode == "partial":
        # BEGIN A06

        # 마지막 특징 추출 블록(layer4)의 가중치 동결 해제
        for param in model.layer4.parameters():
            param.requires_grad = True

        # END A06

        # layer4는 사전학습 가중치를 보존하기 위해 더 작은 학습률 적용
        groups.append({
            "params": model.layer4.parameters(),
            "lr": 0.0001
        })

    # AdamW 최적화 알고리즘 생성
    # 각 파라미터 그룹에 지정된 학습률 사용
    optimizer = torch.optim.AdamW(
        groups, weight_decay=0.0001
    )

    # 학습 데이터 로더: 배치 4개, 셔플 활성화
    train_loader = DataLoader(
        train,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42)
    )

    # 전체 모델을 평가 모드로 설정하여 BN 이동 통계 고정
    model.eval()

    # 최종 분류층만 학습 모드로 변경
    model.fc.train()

    if mode == "partial":

        # 마지막 특징 추출 블록을 학습 모드로 변경
        model.layer4.train()

        # layer4 내부 BatchNorm 계층은 평가 모드로 유지
        # 학습 중 running_mean, running_var가 변경되지 않도록 설정
        for layer in model.layer4.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    # 학습 소요 시간 측정 시작
    start = time.perf_counter()

    # 학습 1 Epoch 수행
    for images, labels in train_loader:

        # 이전 배치에서 계산한 기울기 초기화
        optimizer.zero_grad()

        # 순전파 → 교차 엔트로피 손실 → 역전파
        nn.functional.cross_entropy(
            model(images), labels
        ).backward()

        # 학습 대상 파라미터 갱신
        optimizer.step()

    # 학습 소요 시간 계산
    training_seconds = time.perf_counter() - start

    # 검증을 위한 평가 모드 활성화
    model.eval()

    # 검증 손실, 정답 개수, 전체 샘플 수 초기화
    total_loss, correct, count = 0.0, 0, 0

    # 평가 시 기울기 계산 비활성화
    with torch.no_grad():

        for images, labels in DataLoader(valid, batch_size=4):

            # 모델의 클래스별 예측 점수 계산
            logits = model(images)

            # 배치별 손실을 전체 샘플 기준으로 누적
            total_loss += (
                nn.functional.cross_entropy(logits, labels).item()
                * len(labels)
            )

            # 가장 높은 점수의 클래스를 예측값으로 선택
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)

    # 모델별 학습 조건 및 검증 성능 저장
    rows.append(dict(
        mode=mode,
        pretrained=weights.name,
        image_size=224,
        epochs=1,

        # 실제 학습 가능한 파라미터 개수
        trainable_params=sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),

        # 전체 검증 데이터의 평균 손실과 정확도
        valid_loss=total_loss / count,
        valid_accuracy=correct / count,
        training_seconds=training_seconds
    ))

# 모델별 비교 결과를 터미널에 출력
# 학습 시간은 출력에서 제외하고 소수점 넷째 자리까지 표시
print(
    pd.DataFrame(rows)
    .drop(columns="training_seconds")
    .round(4)
    .to_string(index=False)
)

# 학습 시간을 포함한 전체 비교 결과를 CSV 파일로 저장
pd.DataFrame(rows).to_csv(
    out / "transfer_comparison.csv",
    index=False
)
