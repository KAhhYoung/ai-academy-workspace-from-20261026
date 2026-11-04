
from pathlib import Path
import os
import cv2
import numpy as np

# 실행 위치와 관계없이 현재 Python 파일이 있는 폴더를 기준으로 설정
os.chdir(Path(__file__).resolve().parent)

import csv
import json
from time import perf_counter

Path("outputs").mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # 그래프 창을 띄우지 않고 이미지 파일로 저장
import matplotlib.pyplot as plt

# ① OpenCV는 이미지를 기본적으로 BGR 채널 순서로 읽습니다.
image = cv2.imread("data/scene.png")

# ② BGR → RGB 변환
# OpenCV의 BGR 채널 순서를 Matplotlib이 사용하는 RGB 순서로 변경
# 원본 image는 변경하지 않고 새로운 rgb 배열을 생성합니다.
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ③ 변환하지 않은 BGR 이미지를 Matplotlib으로 출력
# Matplotlib은 RGB로 해석하므로 빨강과 파랑이 뒤바뀌어 표시됩니다.
plt.figure(figsize=(5, 4))
plt.imshow(image)
plt.title("Wrong: BGR interpreted as RGB")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/p01_wrong.png", dpi=120)
plt.close()

# ④ RGB로 변환한 이미지를 정상 출력
# Matplotlib이 RGB 채널을 올바르게 해석하여 실제 색상으로 표시합니다.
plt.figure(figsize=(5, 4))
plt.imshow(rgb)
plt.title("Correct: RGB display")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/p01_correct.png", dpi=120)
plt.close()

# ⑤ BGR 각 채널의 평균값 계산
# axis=(0, 1): 이미지의 높이와 너비 방향으로 평균 계산
# 결과: [Blue 평균, Green 평균, Red 평균]
means = image.mean(axis=(0, 1))

# 채널별 평균값을 CSV 파일로 저장
with open("outputs/p01_channels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["channel", "mean"])
    w.writerows(zip(["B", "G", "R"], means))

print("BGR 채널평균:", np.round(means, 3).tolist())

# ⑥ RGB 이미지의 특정 픽셀 확인
# rgb[30, 30]: 30행, 30열 픽셀의 [R, G, B] 값
print("변환후 빨강:", rgb[30, 30].tolist())
