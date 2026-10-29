"""실습 02: 모양 변환과 장치 이동 후 같은 장치에서 연산합니다."""
import torch
from torch import nn

'''참고사항
GPU: 실제 계산을 수행하는 하드웨어
CUDA: NVIDIA GPU로 계산하도록 지원하는 소프트웨어 기술
PyTorch: CUDA를 활용해 텐서 연산이나 딥러닝 학습을 수행하는 라이브러리
'''

# PyTorch의 난수 생성 시작값(seed)을 42로 고정
torch.manual_seed(42)
# 사용가능하다면 gpu, 그 외 cpu 를 device 로 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 텐서 생성
values = torch.arange(12, dtype=torch.float32)
# >>> H02
# TODO H02: 위 요구사항을 읽고 이 블록을 완성하세요.

# 텐서 shape 변경
matrix = values.reshape(3, 4)
# 텐서 연산
# new_tensor() 를 사용하면 maxtrix 와 같은 자료형과 장치를 사용
shifted = matrix + matrix.new_tensor([1.0, 2.0, 3.0, 4.0])
# <<< H02

# (3,2) 텐서를 gpu 메모리로 이동
X = shifted[:, :2].to(device)

# (3,) 텐서를 gpu 메모리로 이동
# 메모리 크기는 long이 float32보다 큼 / 표현할 수 있는 숫자의 최대 크기(범위)는 float32 가 long 보다 큼
y = torch.tensor([0, 0, 1], dtype=torch.long).to(device)
print("0번째 gpu 이름", torch.cuda.get_device_name(0))
print("gpu 수: ", torch.cuda.device_count())

# 입력특성 2개 → 출력특성 2개인 선형층 모델을 만들고, 파라미터(bias포함)와 버퍼를 gpu 메모리로 이동
# nn.Linear(in_features, out_features, bias=True)
# 입력값 X.shape=(batch_size, in_features)
# 츨력값 y.shape=(batch_size, out_features)
# 가중치 weight.shape=(out_features, in_features)
# 편향 bias.shape=(out_features,)
model = nn.Linear(2, 2).to(device)

# logits.shape=(3, 2)
logits = model(X)
print("행렬:", matrix.tolist())
print("보정:", shifted.tolist())
print("입력/출력:", tuple(X.shape), tuple(logits.shape))


# next(model.parameters()).device는 모델의 첫 번째 파라미터가 있는 장치를 확인하는 코드입니다.
# model.parameters(): 모델의 가중치, 편향 등의 파라미터를 순회하는 반복자
# next(...): 첫 번째 파라미터를 가져옴
# .device: 해당 파라미터가 있는 장치 확인
# 보통 모델 전체를 같은 장치에 두므로 모델의 장치를 확인하는 용도로 사용합니다
print("device 일치:", X.device == y.device == next(model.parameters()).device)
