from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent/'day11_image_classification'
sys.path.insert(0,str(PROJECT))
import torch
from torch import nn
import pandas as pd
from models import SmallCNN, build_model, set_training_mode
from data import load_data

torch.manual_seed(42)
torch.set_num_threads(1)
out=ROOT/'outputs';out.mkdir(exist_ok=True)

mlp=nn.Linear(3*32*32,64)
conv=nn.Conv2d(3,64,3,padding=1)
rows=[]
for name,model in [('Linear',mlp),('Conv2d',conv)]:
    # BEGIN D01
    # 모델의 모든 학습 파라미터(가중치, 편향 등)의 원소 개수를 합산하여 전체 파라미터 수를 계산합니다.
    # model.parameters() : 모델에 등록된 모든 파라미터를 가져옵니다.
    # p.numel()          : 각 파라미터 텐서에 포함된 원소의 개수를 반환합니다.
    # sum(...)           : 각 파라미터의 원소 개수를 모두 더합니다.
    count=sum(p.numel() for p in model.parameters())
    # END D01
    rows.append(dict(model=name,parameters=count))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out/'parameters.csv',index=False)
