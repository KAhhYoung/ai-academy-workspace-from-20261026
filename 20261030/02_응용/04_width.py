from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")

#-----------------------데이터 불러오기 및 학습/검증 데이터 분리-----------------------------#
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
#---------------------------------------------------------------------------------------#



loss_fn = nn.CrossEntropyLoss()

'''TODO A04: 
    width 8과 64만 비교하세요. lr=0.01이고 각 40 epochs입니다.
    서로 다른 폭의 모델은 파라미터 모양이 달라 동일 가중치일 수 없으며, 같은 초기화 seed를 사용합니다.
'''
# nn.Linear 에서의 width: 해당 층의 뉴런 수, 즉 out_features
configs = [{'name': 'width8', 'lr': 0.01, 'width': 8}, {'name': 'width64', 'lr': 0.01, 'width': 64}]

summary = []

for config in configs:
    torch.manual_seed(42)
    width = config.get("width", 32)

    # 하나의 층을 리스트에 담음
    layers = [nn.Linear(2, width)]

    # bn=True 인 경우에 실행: nn.Linear(2,8)-nn.BatchNorm1d(8)-nn.Relu()-nn.Dropout(p=0.0)-nn.Linear(8,8)
    if config.get("bn", False):
        # 특성 width개를 각각 정규화
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # bn=False 인 경우에 실행: nn.Linear(2,8)-nn.Relu()-nn.Dropout(p=0.0)-nn.Linear(8,2)
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])

    # *layers: 리스트 안의 요소들을 풀어서 각각의 인자로 전달. 언패킹(unpacking)
    model = nn.Sequential(*layers).to(device)

    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    # optimizer="SGD"인 경우
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    # optimizer="SGD"가 아닌 경우
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)


    history = []

    # epoch 40번 수행
    for epoch in range(1, 40 + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        history.append([epoch, train_loss, valid_loss, valid_acc])

    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])

    # 검증 손실이 최소인 epoch의 행
    bext_valid_loss_idx = table["valid_loss"].idxmin()
    # 검증 정확도가 최소인 epoch의 행
    bext_valid_acc_idx = table["valid_acc"].idxmax()

    best = table.loc[bext_valid_loss_idx]
    bext_valid_acc_idx = table.loc[bext_valid_acc_idx]
    name = config["name"]
    table.to_csv(out / f"04_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"04_{name}_curves.png", dpi=150)
    plt.close()
    # '검증 손실 최소'를 best 기준으로 하여 출력
    print('검증 손실 최소 기준')
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6), "valid acc:", round(best['valid_acc'], 6))
    # [개인추가]'검증 정확도 최대'를 best 기준으로 하여 출력
    print('검증 정확도 최대 기준')
    print(name, "best epoch:", int(bext_valid_acc_idx["epoch"]), "valid loss:", round(bext_valid_acc_idx["valid_loss"], 6), "best valid acc:", round(bext_valid_acc_idx['valid_acc'], 6))

    #print(name)
    #print(table.loc[table["epoch"] >= 8, ["epoch", "valid_loss"]])
    #print(table.loc[table["epoch"] >= 8, ["epoch", "valid_acc"]])

result = pd.DataFrame(summary)
result.to_csv(out / "04_comparison.csv", index=False)
print("생성: outputs/04_*.csv, *.png / test 미사용")

'''결과 해석
은닉층 너비가 64일 때 검증 손실이 더 낮습니다.
width=64는 입력 데이터의 특성 수가 아니라, 은닉층에서 만들어 내는 특성 수(뉴런 수)입니다.

너비가 커지면서 모델의 표현력이 높아져 데이터의 패턴을 더 잘 학습했을 수 있습니다.
CrossEntropyLoss 기준으로는 검증 데이터의 정답 클래스에 전반적으로 더 높은 확률을 부여했다고 해석할 수 있습니다.

다만 너비를 늘리면 항상 검증 손실이 줄어드는 것은 아닙니다.
이번 실행에서는 64가 더 좋은 결과를 보였으며, 정확도가 얼마나 달라졌는지는 검증 정확도도 함께 확인해야 합니다.


---> 추가로 검증정확도까지 확인함
이번 결과에서는 width64가 검증 손실과 검증 정확도 모두 더 좋습니다.
은닉층 너비 	검증 손실	검증 정확도
8 	          0.099038     	95%
64	          0.007299	    100%


- 정확도 상승: width8은 검증 데이터의 5%를 틀렸고, width64는 모두 맞혔습니다.
- 손실 감소: width64가 정답 클래스에 부여한 확률도 전반적으로 더 좋아졌다는 의미입니다.
즉, 이번 실험에서는 은닉층 뉴런을 늘린 것이 검증 성능 개선으로 이어졌습니다.
다만 검증 정확도 100%가 새로운 데이터에서도 항상 100%라는 뜻은 아닙니다.

또한 두 모델 모두 마지막인 40 epoch에서 최적이었다면, 학습을 더 진행했을 때 추가로 개선되는지도 확인할 수 있습니다.


\outputs\04_width64_curves.png 해석:
width64가 epoch 8에서 정확도 100%에 처음 도달한 뒤에도, 가중치가 갱신되면서 일부 검증 데이터의 예측이 바뀐 것입니다.
정확도가 100%여도 학습은 계속됩니다. CrossEntropyLoss는 정답을 맞혔더라도 정답 확률을 더 높이도록 학습시키기 때문입니다. 이 과정에서 결정 경계 근처의 데이터가 일시적으로 틀릴 수 있습니다.

이 그래프는 검증 손실이 전체적으로 감소하면서 중간중간 출렁이는 형태로 보는 것이 맞습니다.

특히 25 epoch에서는 학습 손실과 검증 손실이 함께 급증했다가 다시 내려갑니다. 검증 손실만 계속 나빠지는 모습과는 달라서, 이 그래프에서 뚜렷한 과적합 징후는 보이지 않습니다.
검증 데이터는 80개이므로 2개만 틀려도 정확도가 100%에서 97.5%로 내려갑니다. 정확도의 출렁임도 이런 작은 오답 수 변화로 이해할 수 있습니다.
따라서 현재 결과는 **“중간에 일시적인 변동이 있지만, 전체적으로 학습·검증 성능이 개선되고 있다”**고 해석하면 됩니다.
'''

