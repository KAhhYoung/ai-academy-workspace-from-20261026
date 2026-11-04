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

x=torch.rand(4,3,32,32)
rows=[]
for channels,kernel in [(8,3),(16,3),(8,5)]:
    # BEGIN D02
    # 2차원 합성곱 계층(Convolution Layer)을 생성합니다.
    # 3                 : 입력 채널 수(in_channels), RGB 이미지이므로 3개 채널
    # channels          : 출력 채널 수(out_channels), 반복문에 따라 8개 또는 16개의 특징 맵 생성
    # kernel            : 커널 크기(kernel_size), 반복문에 따라 3×3 또는 5×5 필터 사용
    # padding=kernel//2 : 커널 크기를 2로 나눈 몫만큼 이미지 가장자리에 패딩 추가
    #                     3×3 커널 → padding=1 (상하좌우 각각 1픽셀)
    #                     5×5 커널 → padding=2 (상하좌우 각각 2픽셀)
    #                     stride=1(기본값)이고 커널 크기가 홀수이므로 입력 이미지의 높이와 너비 유지
    # stride            : 기본값 1, 필터가 한 번에 1픽셀씩 이동
    # 입력 형태         : (4, 3, 32, 32)
    # 출력 형태         : (4, channels, 32, 32)
    layer=nn.Conv2d(3,channels,kernel,padding=kernel//2)
    # END D02
    y=layer(x)
    rows.append(dict(channels=channels,kernel=kernel,shape=str(tuple(y.shape)),parameters=sum(p.numel() for p in layer.parameters())))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out/'kernels.csv',index=False)
