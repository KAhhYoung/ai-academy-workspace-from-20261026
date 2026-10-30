from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "points.csv")

# train / valid 데이터 분리
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.


'''TODO B03:
    학습률 후보 두 개를 리스트로 쓰세요: 0.001, 0.01.
    모델·seed·20 epochs·AdamW·weight_decay=0은 그대로 둡니다.
'''

# 현업에서는 lr 별 변수로 지정하여 실행한다고 하심
# Adam 의 경우 lr=1.0 이라고 하더라도 gradient 크기만큼 그대로 가중치를 이동하진 않음.
learning_rates = [0.001, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.06, 0.08, 0.09, 0.095, 1.0, 1.05]

loss_fn = nn.CrossEntropyLoss()

# lr 변경해보기
for lr in learning_rates:
    # torch.manual_seed(42): “난수를 항상 42로 만든다”는 뜻이 아니라, 난수 생성기의 시작 상태를 42로 정함
    # 매 lr 실험마다 같은 초기 가중치, 같은 난수 상태를 유지하려면 for문 안에 위치해야 함
    torch.manual_seed(42)

    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(X_train), y_train).backward()
        optimizer.step()
        # 갱신이 끝난 모델의 valid loss를 기록합니다. valid로 backward하지 않습니다.
        model.eval()
        with torch.no_grad():
            losses.append(loss_fn(model(X_valid), y_valid).item())
    print("lr:", lr, "처음 5개:", [round(x, 6) for x in losses[:5]])
    print("마지막 valid loss:", round(losses[-1], 6))
