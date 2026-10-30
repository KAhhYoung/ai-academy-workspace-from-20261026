from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import time
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

torch.set_num_threads(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

loss_fn = nn.CrossEntropyLoss()

'''TODO S06:
    baseline, BatchNorm만 추가, weight_decay=0.01만 추가한 세 후보를 만드세요.
    Dropout·BatchNorm·Weight Decay는 목적과 작용 위치가 다른 기법입니다.
'''
configs = [
    {'name': 'baseline', 'bn': False, 'weight_decay': 0.0},
    {'name': 'batchnorm', 'bn': True, 'weight_decay': 0.0},
    {'name': 'weight_decay', 'bn': False, 'weight_decay': 0.01}
]

summary = []

'''레이어 형상
    baseline: nn.Linear(2, 32)-nn.ReLU()-nn.Dropout(p=0.0)-nn.Linear(32,32)-nn.ReLU()-nn.Dropout(p=0.0)-nn.Linear(32,2)
    batchNorm: nn.Linear(2, 32)-nn.BatchNorm1d(32)-nn.ReLU()-nn.Dropout(p=0.0)-nn.Linear(32,32)-nn.BatchNorm1d(32)-nn.ReLU()-nn.Dropout(p=0.0)-nn.Linear(32,2)
    두 모델 모두 형상은 (N, 2) → (N, 32) → (N, 32) → (N, 2)로 동일
'''
for config in configs:
    torch.manual_seed(42)

    # width=32 로 공통
    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]

    # batchNorm=True 인 경우
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])

    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])



    model = nn.Sequential(*layers).to(device)

    # batch_size=32
    batch_size = config.get("batch_size", 32)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))

    # 학습률 조정
    # ReduceLROnPlateau: 관찰하는 지표가 개선되지 않으면 학습률을 줄이는 스케줄러
    # 이번 실습의 관찰지표: valid_loss  / schedule.step(valid_loss) 로 사용가능

    # ReduceLROnPlateau 파라미터 설명
    #   optimizer: 학습률을 조절할 옵티마이저
    #   mode="min": 관찰 지표가 작아져야 개선으로 판단
    #   factor=0.5: 학습률 감소 시 현재 값의 절반으로 변경
    #   patience=2: 개선 없는 epoch를 2번까지 허용하고, 3번 연속이면 학습률 감소
    #   threshold=0.001: 개선으로 인정할 변화량의 기준
    #   threshold_mode="abs": 비율이 아닌 절대 차이로 판단
    #   현재 설정의 개선 조건: 현재 손실 < 기존 best 손실 - 0.001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, 60 + 1):
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        sample_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            sample_loss += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"history_{name}.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*60,
                    "elapsed_seconds": elapsed})
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
