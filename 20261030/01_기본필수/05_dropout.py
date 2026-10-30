import torch
from torch import nn

torch.manual_seed(42)
X = torch.ones(2, 4)
dropout = nn.Dropout(p=0.5)
dropout.train()

'''Dropout 주의사항
    1. activation 원소가 1.0이 아닌 2.0 인 이유: Dropout은 학습 중 살아남은 원소를 1 / (1 - p)배 하기 때문 
    2. 전체 원소의 절반이 제거되지 않은 이유: 각 원소가 독립적으로 50% 확률로 0이 된다는 뜻

        원소가 8개라면:
        - 평균적으로는 4개 제거
        - 실제 한 번 실행에서는 3개, 5개 등이 제거될 수도 있음
          동전을 8번 던졌을 때 앞면이 반드시 4번 나오지는 않는 것과 같습니다.
'''
for i in range(3):
    print("train", i + 1, dropout(X).tolist())

'''TODO B06:
    Dropout을 평가 모드로 바꾸세요. dropout.eval()은 mode만 바꿉니다.
    평가 모드의 Dropout은 입력을 그대로 전달하며, gradient 비활성화와는 별개의 기능입니다.
'''

dropout.eval()

for i in range(3):
    print("eval", i + 1, dropout(X).tolist())
