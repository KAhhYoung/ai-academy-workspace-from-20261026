"""실습 04: 2 → 8 → 2 신경망의 순전파를 확인합니다."""
import torch
from torch import nn

torch.manual_seed(42)

'''TODO B04: 신경망의 '순전파' 학습 --> 이 시점에는 학습하지 않았습니다. 따라서, logits는 원시 점수입니다. 
    Sequential의 마지막 Linear를 수정해 출력 클래스 수가 2가 되게 하세요.
    첫 Linear(2, 8)와 ReLU는 그대로 두세요. 모델 정의는 미리 제공했습니다.
'''
# nn.ReLu()는 입력텐서의 shape 를 그대로 유지
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
# <<< B04
# X.shape=(4, 2)
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0], [2.0, 0.0]])

# logits.shape=(4, 2)
logits = model(X)
print("입력:", tuple(X.shape))
print("출력:", tuple(logits.shape))

# type(p) : torch.nn.parameter.Parameter
# Parameter 는 Tensor 의 하위 클래스
# 0: nn.Linear / 1: nn.Relu / 2: nn.Linear
# weight.shape  # (out_features, in_features)  / 입력을 열벡터로    / 0번째 Linear의 shape=(8,2) , 2번째 Linear의 shape=(2,8)
# bias.shape    # (out_features,)
for name, p in model.named_parameters():        #print("파라미터 수:", sum(p.numel() for p in model.parameters()))
    print(f"{name}: {p.numel()}개")
