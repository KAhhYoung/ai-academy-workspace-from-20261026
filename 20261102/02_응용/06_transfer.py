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

from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
train = datasets.ImageFolder(root / "data" / "train", transform=weights.transforms())
valid = datasets.ImageFolder(root / "data" / "valid", transform=weights.transforms())
# 1 epoch는 학습 경로 점검용이다. 이 숫자로 일반화 성능의 우열을 단정하지 않는다.
rows = []
for mode in ["head", "partial"]:
    torch.manual_seed(42)
    model = resnet18(weights=weights)  # 가중치가 없으면 공식 다운로드. 무작위 초기화로 대체하지 않는다.

    # Resnet18 backbone 동결(freeze)
    for param in model.parameters():
        param.requires_grad = False

    # 새로 만든 fc는 기본적으로 requires_grad=True
    model.fc = nn.Linear(model.fc.in_features, 2)
    groups = [{"params": model.fc.parameters(), "lr": 0.001}]

    # 부분 미세조정 모드에서는 layer4도 학습할 수 있도록 동결을 해제한다.
    if mode == "partial":
        # TODO A06: partial일 때만 layer4의 미분을 켠다.
        for param in model.layer4.parameters():
            param.requires_grad = True

        groups.append({"params": model.layer4.parameters(), "lr": 0.0001})
    
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0001)
    train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(42))

    # 기존 특징 추출부의 BatchNorm 통계 갱신을 막고, Dropout을 비활성화하면서 분류 헤드를 학습할 때 사용하는 패턴
    # 기존 특징 추출부: 평가모드 / fc: 학습 모드
    model.eval()

    # fc는 nn.Linear이므로 train/eval에 따른 계산 차이는 없지만, 분류 헤드를 학습 모드로 설정한다는 의도를 명시한다.
    model.fc.train()

    # 부분 미세조정 모드에서는 layer4와 그 하위 층을 학습 모드로 전환한다.
    # layer4에서도 BatchNorm 층은 평가모드로 설정하는 이유(=running_mean,running_var 갱신안함):
    #   -작은 데이터·배치로 미세조정할 때, 사전 학습된 BN 통계를 유지해 학습을 안정시키려는 선택
    #   -필수 규칙은 아님. 새 데이터가 충분하고 배치가 크거나, ImageNet과 데이터 특성이 크게 다르면 BN 통계도 갱신하는 방법을 검토할 수 있음
    if mode == "partial":
        model.layer4.train()
        for layer in model.layer4.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    start = time.perf_counter()
    for images, labels in train_loader:
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
    training_seconds = time.perf_counter() - start

    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in DataLoader(valid, batch_size=4):
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels).item()*len(labels)
            correct += (logits.argmax(1)==labels).sum().item()
            count += len(labels)
    rows.append(dict(mode=mode,
                     pretrained=weights.name,
                     image_size=224,
                     epochs=1,
                     trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     valid_loss=total_loss/count,
                     valid_accuracy=correct/count,
                     training_seconds=training_seconds
    ))
    
print(pd.DataFrame(rows).drop(columns="training_seconds").round(4).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "transfer_comparison.csv", index=False)
