"""실습 03 · 클래스 정의 없이 Sequential로 작은 CNN을 연결한다."""
import torch
from torch import nn

torch.manual_seed(42)
x = torch.rand(4, 3, 32, 32)
# MaxPool2d(2)는 2×2에서 최댓값을 선택한다. 기본 stride도 2이다.
# BEGIN B03
pool = nn.MaxPool2d(2)  # pool크기를 2로 설정하여 출력 높이·너비를 32의 절반으로 만들어요
# END B03
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), pool,
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
)
for layer in model:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
