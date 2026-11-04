import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)

x = torch.randn(2,5,16)     # (배치크기, 문장마다 토큰 수, 토큰 하나의 벡터 차원)
attention = nn.MultiheadAttention(embed_dim=16,         # 토큰 하나를 표현하는 벡터의 차원
                                  num_heads=4,          # attention을 병렬로 계산하는 head 수
                                  dropout=0.0,          # 학습 중 attention 가중치에 적용할 드롭아웃 확률
                                  batch_first=True      # 입출력 텐서에서 배치 차원을 맨 앞에 배치
)
# nn.MultiheadAttention 내부에서 학습 가능한 선형 변환으로 Q, K, V를 생성

# TODO B05: 4개 head의 가중치를 평균내지 않고 각각 받는다.
# query/key/value에는 같은 x를 넣는다. average_attn_weights=False를 지정한다.
output, weights = attention(x,x,x,average_attn_weights=False)

print("입력:", tuple(x.shape))
print("출력:", tuple(output.shape))
print("head별 weight:", tuple(weights.shape))
print("각 행의 합:", weights[0,0].sum(dim=-1).detach().round(decimals=4))
