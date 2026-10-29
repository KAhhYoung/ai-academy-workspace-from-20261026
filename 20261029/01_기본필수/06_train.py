"""실습 06: 작은 정상 자료로 학습 순서만 익힙니다. 여기는 train 오차 확인용입니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.tensor([[-2., -1.], [-1., -1.], [-1., 0.], [0., -1.],     # X.shape=(8,2)
                  [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)        # y.shape=(8,)

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("학습 전 loss:", round(loss_fn(model(X), y).item(), 6))

# model.train(): 모델을 학습 모드로 설정
# model.train() 자체가 학습을 실행하거나 가중치를 갱신하지는 않습니다. 실제 학습에는 순전파, loss.backward(), optimizer.step() 등이 필요합니다.
# model.eval() 은 평가 모드이므로 평가 후 다시 학습한다면 model.train() 호출해야 함
model.train()

'''전체적인 흐름
    optimizer.zero_grad()                       # 이전 기울기 초기화

    logits = model(x)                           # 순전파
    loss = loss_fn(logits, y)                   # 손실 계산
    loss.backward()                             # 기울기 계산

    for name, p in model.named_parameters():
        print(name, p.grad)                     # 각 파라미터에 대한 손실의 기울기

    optimizer.step()                            # 기울기로 파라미터 갱신
'''


for epoch in range(80):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    
    '''TODO B06:
        현재 loss의 gradient를 계산하세요.
        파라미터 값은 아직 바뀌지 않으며 .grad에 기울기가 들어갑니다.
    '''
    # loss.backward()를 호출하면 손실을 각 모델 파라미터로 미분한 기울기가 계산됩니다.
    # 결과는 각 파라미터의 .grad 에 저장됩니다.
    loss.backward()

    '''TODO B07:
        현재 gradient를 사용해 파라미터를 한 번 갱신하세요.
        optimizer의 step 메서드를 호출합니다.
    '''
    optimizer.step()
    
model.eval()

'''자동 미분의 연산 기록 조건 (아래 조건을 모두 만족할 때):
    1. 기울기 추적 모드가 활성화되어 있음.
       예: torch.no_grad() 안에서는 비활성화됨. 
    2. 연산에 참여하는 텐서 중 하나 이상이 requires_grad=True임.
    3. 해당 연산이 자동 미분을 지원함.
'''
with torch.no_grad():
    logits = model(X)
    print("학습 후 loss:", round(loss_fn(logits, y).item(), 6))
    print("train 정확도:", round((logits.argmax(dim=1) == y).float().mean().item(), 4))
print("주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.")
