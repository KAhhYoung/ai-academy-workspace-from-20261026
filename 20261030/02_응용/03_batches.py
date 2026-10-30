from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib

# Matplotlib이 그래프를 그릴 때 화면에 창을 띄우지 않고, 이미지 파일로 저장할 수 있는 백엔드를 사용하도록 설정
# pyplot을 import하기 전에 설정
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)

# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")

#---------------------데이터 불러오기---------------------#
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
#-------------------------------------------------------#

loss_fn = nn.CrossEntropyLoss()


'''TODO A03: 
    batch_size 후보를 [8, 32]로 두세요. 총 train 행은 240건입니다.
    drop_last=False이므로 32인 경우 마지막 배치는 16건입니다.
'''
batch_sizes = [8, 32]
rows = []
for size in batch_sizes:
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=size, shuffle=True)
    counts = [len(yb) for xb, yb in loader]

    rows.append([size, len(counts), counts[-1], sum(counts)])

    print("batch:", size, "iterations:", len(counts), "마지막:", counts[-1], "처리 수:", sum(counts))

df = pd.DataFrame(rows, columns=["batch_size", "iterations", "last_batch", "samples"])
df.to_csv(out / "03_batches.csv", index=False)
print(df)
