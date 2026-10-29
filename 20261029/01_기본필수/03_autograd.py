"""실습 03: x=3에서 y=x²+2x+1과 기울기를 계산합니다."""
import torch
x = torch.tensor(3.0, requires_grad=True)

# TODO B03: 계산 그래프가 유지되도록 Tensor 연산으로 식을 작성하세요.
# x를 Python 숫자로 변환하지 마세요.
y = x ** 2 + 2 * x + 1

# backward는 y 값을 바꾸지 않고 x.grad에 기울기를 계산합니다.
print("역전파 전 y:", y.item())
y.backward()
print("역전파 후 y:", y.item())

# 역전파로 x.grad에 저장된 기울기(dy/dx)를 출력
# 기울기: 함수값이 증가하는 방향과 변화율을 나타내는 값
# 손실을 줄이도록 파라미터 업데이트: x = x - 학습률 * x.grad
print("dy/dx:", x.grad.item())
