"""실습 05: 정답 인덱스와 logits로 손실을 계산합니다."""
import torch
from torch import nn

logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
target = torch.tensor([0, 1], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()     # input: Tensor, target: Tensor -> Tensor

'''TODO B05:
    loss_fn에 logits와 target을 순서대로 넣어 loss를 계산하세요.
    logits에 softmax를 먼저 적용하지 마세요.
'''
loss = loss_fn(logits, target)
print(loss.shape)   # 기본적으로 배치의 손실을 평균 내어 **스칼라(0차원 텐서)**를 반환하므로 shape가 []입니다.

'''nn.CrossEntropyLoss() 과정'''
# 각 샘플의 클래스 점수 2개에 softmax를 적용해 확률을 구합니다.
# 각 샘플에서 정답 클래스의 확률을 선택합니다.
# 그 확률에 -log()를 적용해 샘플별 손실을 구합니다.
# 샘플 2개의 손실을 평균 냅니다.

print("logits:", tuple(logits.shape))
print("target:", tuple(target.shape), target.dtype)
# loss.item(): 원소가 1개일 때만 사용할 수 있습니다.
print("loss:", round(loss.item(), 6))
