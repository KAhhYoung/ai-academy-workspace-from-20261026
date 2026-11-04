from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

image = cv2.imread("noisy.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# >>> 해설 B03
# 양의 홀수 커널을 사용합니다. sigmaX=0은 커널 크기에서 표준편차를 계산하도록 합니다. Edge 개수는 품질 점수가 아닙니다.
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blur, 80, 160) # 경계선을 검출하기 위한 함수에요
# <<< 해설 B03
cv2.imwrite("p03_blur.png", blur)
cv2.imwrite("p03_edge.png", edge)
print("설정: Gaussian (5,5), sigmaX=0, Canny 80/160")
print("Shape:", blur.shape, edge.shape)
print("Edge 픽셀:", int(np.count_nonzero(edge)))
