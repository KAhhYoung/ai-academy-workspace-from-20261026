import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)

x = torch.randn(2,5,16)
attention = nn.MultiheadAttention(16,4,dropout=0.0,batch_first=True)
# 해설 B05: 4개 head의 가중치를 평균내지 않고 각각 받는다.
# query/key/value에는 같은 x를 넣는다. average_attn_weights=False를 지정한다.
# BEGIN B05
# Self-Attention 수행: 입력 x를 Query(Q), Key(K), Value(V)로 모두 사용합니다.
# 각 토큰이 다른 토큰을 얼마나 참고해야 하는지 Attention 가중치를 계산하고,
# 그 가중치를 이용해 문맥 정보가 반영된 새로운 토큰 표현(output)을 생성합니다.
#
# attention(x, x, x)
#   - 첫 번째 x (Query): 현재 토큰이 어떤 정보를 찾고 싶은가?
#   - 두 번째 x (Key)  : 각 토큰이 어떤 정보를 가지고 있는가?
#   - 세 번째 x (Value): 실제로 가져와 조합할 정보는 무엇인가?
#
# average_attn_weights=False
#   - 여러 Attention Head의 가중치를 평균 내지 않고 각각 반환합니다.
#   - Head마다 어떤 토큰에 집중하는지 개별적으로 분석할 수 있습니다.
#
# output : Attention 결과로 생성된 문맥이 반영된 토큰 벡터
# weights: 각 Attention Head가 토큰들에 부여한 Attention 가중치
output, weights = attention(x, x, x, average_attn_weights=False)
#   attention()을 하는 이유! 핵심 목적은 각 토큰이 다른 토큰과의 관계를 파악하여 문맥을 반영한 새로운 표현을 만들기 위해서
# END B05
print("입력:", tuple(x.shape))
print("출력:", tuple(output.shape))
print("head별 weight:", tuple(weights.shape))
print("각 행의 합:", weights[0,0].sum(dim=-1).detach().round(decimals=4))
