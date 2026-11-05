"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torchvision.models import ResNet18_Weights
from PIL import Image

torch.set_num_threads(1)
torch.manual_seed(14)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.

# Compose는 각 변환의 결과를 다음 변환에 전달
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

weights = ResNet18_Weights.IMAGENET1K_V1  # Enum·변환 생성은 다운로드하지 않음

# weights.transforms()는 해당 사전 학습 가중치에 맞는 이미지 전처리 객체를 반환
# 사전 학습 때 모델이 보았던 입력과 비슷한 형태로 이미지를 제공해야, 학습된 특징을 잘 활용할 수 있음
# TODO A02: Weight가 제공하는 transforms()를 가져오세요.
official = weights.transforms()

# resize_size = [256]: 가로세로 비율을 유지하면서 짧은 변을 256픽셀로 맞춥니다. 256×256으로 강제 변환하는 것은 아닙니다.
# crop_size = [224]: 크기 조정 후 중앙의 224×224 영역을 잘라냅니다.
print("공식 Resize/Crop:", official.resize_size,official.crop_size)

# Image.open()은 원본 파일의 색상 모드를 유지하므로, 항상 RGB는 아닙니다. "RGB", "RGBA", "L", "P" 등
image = Image.open(sorted((DATA/"valid"/"circle").glob("*.png"))[0]).convert("RGB")

source_tensor = transform(image)
print("소형 source 입력:",tuple(source_tensor.shape),"범위 [0,1]")  
print("공식 ResNet 입력:",tuple(official(image).shape))     

source_mean = source_tensor.mean(dim=(1, 2))
source_std = source_tensor.std(dim=(1, 2), correction=0)    # correction은 표준편차 계산에서 분모를 얼마나 줄일지 정하는 인자입니다.
print("소형 mean/std:", source_mean,source_std)
# ImageNet 기준으로 미리 정해진 RGB 채널별 정규화 값
print("공식 mean/std:", official.mean,official.std)

print("작은 source와 ImageNet의 Normalize를 서로 섞지 않습니다.")


'''
[결론]
-resize 차이
 1) 소형 source: (3, 32, 32)
 2) 공식 ResNet: (3, 224, 224)
 채널은 3개로 동일하나 소형 source 의 입력이미지 크기가 더 작음

-normalize 차이:
 소형 mean/std 는 이미지 한 장에 대한 평균/표준편차 이므로
 official mean/std 와 비교하기에 무리가 있음
'''