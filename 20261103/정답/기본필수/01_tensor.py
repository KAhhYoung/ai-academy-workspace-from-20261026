"""실습 01 · RGB 이미지의 N,C,H,W를 확인한다. 실제 사진 대신 정상 Tensor 4장."""
import torch

torch.manual_seed(42)
# 해설 B01: rand의 네 크기는 이미지 수, 채널, 높이, 너비이다.
# 필수: 각 차원의 크기. 선택: dtype, device. 기본 float32·CPU를 사용한다.
# BEGIN B01
# 0 이상 1 미만의 균등분포 난수로 구성된 가상 이미지 데이터 4개를 생성합니다.
# 4  : 배치 크기(Batch Size), 이미지 4개
# 3  : 채널 수(Channel), RGB 색상 채널 3개
# 32 : 이미지 높이(Height), 32픽셀
# 32 : 이미지 너비(Width), 32픽셀
# 텐서 형태: (N, C, H, W) = (4, 3, 32, 32)
images = torch.rand(4, 3, 32, 32)
# END B01
print("shape:", tuple(images.shape))
print("dtype:", images.dtype)
print("device:", images.device)
print("첫 이미지:", tuple(images[0].shape))
