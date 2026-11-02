
from pathlib import Path
import sys

# 현재 실습 폴더 및 공통 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'day11_image_classification'
sys.path.insert(0, str(PROJECT))

import torch
from torch import nn
import pandas as pd
from models import SmallCNN, build_model, set_training_mode
from data import load_data

# 난수 고정 및 CPU 연산 스레드 수 제한
torch.manual_seed(42)
torch.set_num_threads(1)

# 결과 저장 폴더 생성
out = ROOT / 'outputs'
out.mkdir(exist_ok=True)

# 모델별 실험 설정: (모델 종류, 모델 폭, Epoch)
# small   : 사전학습 없이 작은 CNN을 처음부터 학습
# head    : 사전학습된 ResNet의 최종 분류층(fc)만 학습
# partial : 사전학습된 ResNet의 마지막 블록(layer4)과 fc를 학습
# 전이 모델은 ImageNet 사전학습 가중치를 사용하며, 없으면 다운로드
# BEGIN D06
SETTINGS = [    # 설정들을 그룹화 하는거예요
    ('small', 16, 1),
    ('head', 16, 1),
    ('partial', 16, 1)
]
# END D06

from models import set_training_mode
from train import evaluate
import copy
import time

rows = []  # 모델별 최종 비교 결과 저장

# 설정된 3가지 모델을 순서대로 학습 및 평가
for kind, width, epochs in SETTINGS:

    # 각 모델 실험 시작 시 난수 시드 재설정
    torch.manual_seed(42)

    # 데이터 로드: SmallCNN은 배치 16, ResNet은 배치 4
    train, _, valid, loader, train_check, valid_loader = load_data(
        kind, batch_size=16 if kind == 'small' else 4
    )

    # 모델 생성 (전이 모델은 사전학습 가중치 사용)
    model = build_model(kind, width=width)

    # 모델 종류에 따라 학습할 파라미터 및 학습률 설정
    if kind == 'partial':
        # 마지막 블록과 분류층을 서로 다른 학습률로 미세조정
        groups = [
            {'params': model.fc.parameters(), 'lr': 0.001},
            {'params': model.layer4.parameters(), 'lr': 0.0001}
        ]
    elif kind == 'head':
        # 최종 분류층(fc)만 학습
        groups = model.fc.parameters()
    else:
        # SmallCNN은 전체 파라미터 학습
        groups = model.parameters()

    # AdamW 최적화 알고리즘 생성
    optimizer = torch.optim.AdamW(
        groups,
        lr=0.01 if kind == 'small' else 0.001,
        weight_decay=0.0001
    )

    # 학습 기록, 최적 검증 손실 및 실행 시간 초기화
    history = []
    best = float('inf')
    start = time.perf_counter()

    # 지정된 Epoch만큼 모델 학습
    for epoch in range(1, epochs + 1):

        # 모델 유형에 맞게 학습 모드 및 동결 계층 설정
        set_training_mode(model, kind)

        for x, y in loader:
            # 이전 기울기 초기화
            optimizer.zero_grad(set_to_none=True)

            # 순전파 → 손실 계산 → 역전파 → 가중치 갱신
            nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()

        # 학습 및 검증 데이터의 손실과 정확도 평가
        tl, ta, _, _ = evaluate(
            model, train_check, torch.device('cpu')
        )
        vl, va, _, _ = evaluate(
            model, valid_loader, torch.device('cpu')
        )

        # Epoch별 학습/검증 성능 기록
        history.append(dict(
            epoch=epoch,
            train_loss=tl,
            valid_loss=vl,
            train_accuracy=ta,
            valid_accuracy=va
        ))

        # 검증 손실이 가장 낮은 모델의 가중치 저장
        if vl < best:
            best = vl
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # 전체 학습 및 평가 소요 시간 측정
    seconds = time.perf_counter() - start

    # 검증 손실 기준 최적 모델 복원
    model.load_state_dict(best_state)

    # 최적 모델의 검증 손실 및 정확도 재계산
    vl, va, _, _ = evaluate(
        model, valid_loader, torch.device('cpu')
    )

    # 모델 이름 생성 (예: small_w16, head_w16)
    name = f'{kind}_w{width}'

    # 모델별 Epoch 학습 이력을 CSV로 저장
    pd.DataFrame(history).to_csv(
        out / f'history_{name}.csv', index=False
    )

    # 모델별 최종 비교 정보 저장
    rows.append(dict(
        model=name,

        # requires_grad=True인 학습 가능 파라미터 수
        trainable_params=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),

        # SmallCNN은 32×32, ResNet은 224×224 이미지 사용
        image_size=32 if kind == 'small' else 224,

        epochs=epochs,
        best_epoch=best_epoch,
        valid_loss=vl,
        valid_accuracy=va,
        training_seconds=seconds,

        # 사전학습 가중치 사용 여부
        source_weights='none' if kind == 'small' else 'IMAGENET1K_V1'
    ))

    # 현재 모델의 최종 검증 성능 출력
    print(
        name, 'best', best_epoch,
        'valid loss', round(vl, 4),
        'accuracy', round(va, 4)
    )

# 3개 모델의 성능 비교 결과를 CSV 파일로 저장
pd.DataFrame(rows).to_csv(
    out / 'comparison.csv', index=False
)

# 모델별 이미지 크기, 배치 크기, 학습률이 달라
# 순수한 모델 구조 비교가 아닌 학습 전략 전체의 비교임을 안내
print(
    'small32/ResNet224·배치16/4·학습률의 차이가 있으므로 '
    '학습 전략 전체의 비교이다.'
)
