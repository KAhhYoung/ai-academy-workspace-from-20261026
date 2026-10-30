from pathlib import Path
import copy

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 그래프 창을 띄우지 않고 이미지 파일로 저장합니다.
import matplotlib.pyplot as plt


# CPU 연산에 사용하는 스레드를 1개로 제한합니다.
# CPU/GPU 선택과는 별개의 설정입니다.
torch.set_num_threads(1)

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(base / "data" / "points.csv")

# train은 파라미터 학습에, valid는 성능 비교와 최적 epoch 선택에 사용합니다.
# test 데이터는 이 실습에서 사용하지 않습니다.
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

X_train = torch.tensor(
    train[["x1", "x2"]].values, dtype=torch.float32
)
y_train = torch.tensor(
    train["label"].values, dtype=torch.long
)
X_valid = torch.tensor(
    valid[["x1", "x2"]].values, dtype=torch.float32
).to(device)
y_valid = torch.tensor(
    valid["label"].values, dtype=torch.long
).to(device)

# CrossEntropyLoss에는 softmax를 적용하지 않은 logits와 정답 인덱스를 넣습니다.
loss_fn = nn.CrossEntropyLoss()

# SGD와 AdamW를 비교하고, AdamW에서는 Dropout의 유무도 비교합니다. (여기 중요! 실무에서 이렇게 자주 쓰여요!)
# 같은 lr을 사용하지만, 각 optimizer에 가장 적합한 lr이라는 의미는 아닙니다.
configs = [
    {"name": "sgd", "optimizer": "SGD", "lr": 0.01},
    {"name": "adamw", "lr": 0.01},
    {"name": "adamw_dropout", "lr": 0.01, "dropout": 0.3},
]
summary = []

for config in configs:
    # 후보마다 seed를 다시 설정해 같은 구조의 모델은 같은 초기 가중치로 시작합니다.
    torch.manual_seed(42)

    # 설정에 width가 없으면 기본값 32를 사용합니다.
    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]

    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, width),
    ])

    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, 2),
    ])

    model = nn.Sequential(*layers).to(device)

    # 후보마다 같은 seed의 독립 Generator를 만들어 배치 순서를 동일하게 맞춥니다. (여기 중요!)
    # epoch가 바뀌면 새 순서로 섞이지만, 후보끼리는 같은 순서로 진행됩니다.
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(43),
        num_workers=0,
    )

    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.0),
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.0,
        )

    history = []    #   통계 기록용 리스트: epoch, train_loss_eval, valid_loss, valid_acc
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    # 이 코드는 Early Stopping 없이 40 epoch를 모두 학습한 뒤,
    # 그중 valid loss가 가장 낮았던 모델을 선택합니다.
    for epoch in range(1, 41):
        # 학습 모드: Dropout을 활성화하고, BatchNorm이 있다면 배치 통계를 사용합니다.
        model.train()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # 이전 기울기 제거 → 순전파·손실 계산 → 역전파 → 파라미터 갱신
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 평가 모드에서는 Dropout이 꺼집니다.
        # train과 valid를 모두 같은 평가 조건의 모델로 측정합니다.
        model.eval()

        # no_grad는 기울기 계산용 그래프 기록을 끕니다.
        # eval과 역할이 다르므로 평가할 때 함께 사용합니다.
        with torch.no_grad():
            train_loss = loss_fn(
                model(X_train.to(device)), y_train.to(device)
            ).item()

            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()

            # 샘플별로 점수가 가장 높은 클래스와 정답을 비교해 정확도를 구합니다.
            # 클래스 선택만 필요하므로 softmax 없이 argmax를 사용합니다.
            valid_acc = (
                (logits.argmax(1) == y_valid).float().mean().item()
            )

            # 최고 기록을 갱신했을 때만 저장합니다. 동률이면 이전 epoch를 유지합니다.
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch

                # 모델 상태를 별도로 복사해 이후 학습으로 저장값이 바뀌지 않게 합니다.
                # 가중치·편향과 BatchNorm의 누적 통계 등이 포함됩니다.
                best_state = copy.deepcopy(model.state_dict())

        history.append([epoch, train_loss, valid_loss, valid_acc])

    table = pd.DataFrame(
        history,
        columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"],
    )
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]

    table.to_csv(out / f"07_{name}_history.csv", index=False)

    summary.append({
        **config,
        "best_epoch": int(best["epoch"]),
        "best_valid_loss": float(best["valid_loss"]),
        "valid_acc_at_best": float(best["valid_acc"]),
        "last_train_loss_eval": train_loss,
        "last_valid_loss": valid_loss,
    })

    # 현재 모델은 마지막 epoch 상태이므로, 저장해 둔 최적 epoch 상태로 복원합니다.
    # 위 summary의 last_* 값은 복원 전 마지막 epoch의 성능입니다.
    model.load_state_dict(best_state)
    model.eval()    # Dropout을 끄고 BatchNorm이 있다면 누적 통계를 사용합니다.

    with torch.no_grad():
        best_pred = model(X_valid).argmax(1).cpu()
        restored_loss = loss_fn(model(X_valid), y_valid).item()

    labels = y_valid.cpu()
    class_rows = []

    for label in (0, 1):
        # 해당 클래스가 실제 정답인 샘플만 고릅니다.
        mask = labels == label
        support = int(mask.sum())

        # 그중 해당 클래스로 올바르게 예측한 샘플 수입니다.
        correct = int(((best_pred == label) & mask).sum())

        # Recall = 해당 클래스를 맞힌 수 / 실제 해당 클래스의 전체 수
        # 실제 표본이 없으면 계산할 수 없으므로 None으로 기록합니다.
        recall = correct / support if support else None

        class_rows.append({
            "class": label,
            "support": support,
            "correct": correct,
            "recall": recall,
        })

    pd.DataFrame(class_rows).to_csv(
        out / f"07_{name}_class_metrics.csv", index=False
    )

    pd.DataFrame({
        "id": valid["id"].tolist(),
        "actual": labels.tolist(),
        "predicted": best_pred.tolist(),
    }).to_csv(out / f"07_{name}_predictions.csv", index=False)

    # 복원 후 손실이 저장 당시의 최저 손실과 일치하는지 확인합니다.
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = (
        abs(restored_loss - best_loss) < 1e-6
    )

    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best_epoch, linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()

    print(
        name,
        "best epoch:", best_epoch,
        "best valid loss:", round(best_loss, 6),
    )

result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")