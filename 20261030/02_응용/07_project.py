from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib

# 창을 띄우지 않고 PNG에 저장합니다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# 사용할 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 불러오기 및 학습/검증 데이터 분리 ---------------- #
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
# ------------------------------------------------------------------ #

# 손실함수 정의
loss_fn = nn.CrossEntropyLoss()

# 비교 대상: 각 실험에서 비교 항목 외의 조건은 동일하게 유지
#   1) 옵티마이저 비교: SGD vs AdamW
#   2) Dropout 비교: 미적용(p=0.0) vs 적용(p=0.3)
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01, 'dropout': 0.0},
           {'name': 'adamw', 'optimizer': 'adamw', 'lr': 0.01, 'dropout': 0.0},
           {'name': 'adamw_dropout', 'optimizer': 'adamw', 'lr': 0.01, 'dropout': 0.3}
]
summary = []

'''모델의 텐서 형상 변화: (N, 2) → (N, 32) → (N, 2)
        nn.Linear(2, 32)
        nn.ReLU()
        nn.Dropout(p=config.get("dropout", 0.0))
        nn.Linear(32,2)
'''
for config in configs:
    torch.manual_seed(42)

    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]

    # ------모든 모델이 bn=False 이므로 실행되지 않는 코드
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout")), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # -----------------------------------------------

    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout")), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)

    # 배치를 제공할 DataLoader를 설정
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    # optimizer="SGD"인 경우 (momentum, weight_decay 적용 안함)
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
        
    # optimizer="AdamW"인 경우 (momentum, weight_decay 적용 안함)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    # epoch=40 으로 설정
    for epoch in range(1, 40 + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            
            '''TODO A07:
                loss.backward() 다음 optimizer.step()을 호출하세요.
                같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분합니다.
            '''
            loss.backward()
            optimizer.step()

        '''TODO A08:
            같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
            epoch 종료 후 model.eval()을 호출하세요.
            그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않습니다.
        '''
        model.eval()
        with torch.no_grad():
            # 평가 모드에서 전체 학습 데이터의 평균 손실 계산
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)

            # 평가 모드에서 전체 검증 데이터의 평균 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 전체 검증 데이터의 정확도 계산
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()

            '''TODO A09: 
                최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
                동률은 먼저 발견한 epoch 유지. copy.deepcopy(model.state_dict()) 사용.
                model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 됩니다.
                최적 loss·epoch와 독립된 state_dict 복사본을 보관하세요.
            '''
            # 검증 손실이 개선되면 최적 손실, epoch, 모델 상태 갱신
            # model.state_dict(): 모델의 학습 파라미터와 버퍼를 이름별로 담아 반환하는 딕셔너리 / 주로 모델 저장과 복원에 사용
            # valid_loss, epoch 는 별도로 저장 필요
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        # 매 epoch의 학습 손실, 검증 손실 및 검증 정확도 기록
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # epoch별 학습·검증 기록을 데이터프레임으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])

    # 검증 손실이 최소인 epoch의 기록 선택
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]

    table.to_csv(out / f"07_{name}_history.csv", index=False)

    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    
    # 학습이 끝난 뒤 최적 상태로 복원
    model.load_state_dict(best_state)

    # 학습 종료 후 복원한 최적 모델로 전체 검증 데이터의 예측값과 손실 계산
    model.eval()
    with torch.no_grad():
        # X_valid 가 GPU 에 있으므로 GPU에서 예측과 argmax를 수행한 후, 결과를 CPU로 옮깁니다.
        logits = model(X_valid)

        # 전체 검증 데이터의 예측 클래스 번호
        best_pred = logits.argmax(1).cpu()

        # 전체 검증 데이터의 평균 손실 계산
        restored_loss = loss_fn(logits, y_valid).item()

        # 전체 검증 데이터의 정확도 계산
        restored_acc = (logits.argmax(1) == y_valid).float().mean().item()

    labels = y_valid.cpu()
    class_rows = []

    # 실제 정답 클래스별 표본 수, 맞힌 수, 재현율 기록
    for label in (0, 1):
        '''TODO A10: 클래스별 실제 표본 수·정답 수·Recall을 기록하세요.
            실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
            전체 accuracy나 '예측 label의 수'로 나누지 않습니다. 0건이면 recall=None.
        '''
        # 실제 정답이 label인 데이터 선택
        mask = labels == label

        # 해당 클래스의 실제 표본 수
        support = int(mask.sum())

        # 해당 클래스의 표본 중 올바르게 예측한 수
        correct = int(((best_pred == label) & mask).sum())

        # Recall = 해당 클래스를 맞힌 수 / 실제 해당 클래스의 전체 수
        # 해당 클래스의 실제 표본이 없으면 계산할 수 없으므로 None으로 기록합니다.
        recall = correct / support if support else None

        class_rows.append({
            "class": label,
            "support": support,
            "correct": correct,
            "recall": recall,
        })

    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
    print(config["name"], "restored_valid_acc:", round(restored_acc, 6))
result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print(result)

print("생성: outputs/07_*.csv, *.png / test 미사용")

