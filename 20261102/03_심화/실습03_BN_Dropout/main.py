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

model=SmallCNN(batchnorm=True,dropout=0.3)
x=torch.rand(4,3,32,32)
model.train()

# nn.BatchNorm2d(16)
# model.features[1].running_mean: 해당 BatchNorm2d 층에 들어온 각 배치의 채널별 평균을, 여러 배치에 걸쳐 지수 이동 평균한 값
before=model.features[1].running_mean.clone()
y1,y2=model(x),model(x)
after=model.features[1].running_mean.clone()

model.eval()

with torch.no_grad():
    e1,e2=model(x),model(x)
print('train 출력 동일:',torch.equal(y1,y2))
print('BN 통계 변화:',not torch.equal(before,after))
print('eval 출력 동일:',torch.equal(e1,e2))
print('shape:',tuple(e1.shape))

# 터미널에 출력한 실습 결과를 텍스트 파일로 저장합니다.
results = [
    f"train 출력 동일: {torch.equal(y1, y2)}",
    f"BN 통계 변화: {not torch.equal(before, after)}",
    f"eval 출력 동일: {torch.equal(e1, e2)}",
    f"shape: {tuple(e1.shape)}"
]

# outputs/result.txt 파일 생성 및 저장
result_path = out / "result.txt"
result_path.write_text("\n".join(results) + "\n", encoding="utf-8")

print(f"결과 파일 저장 완료: {result_path}")