from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

image = cv2.imread("scene.png")
# >>> 해설 B02
# BGR→Gray는 채널을 1개로 줄이며, BGR→RGB는 채널 순서를 바꿉니다. 원본 비율을 무시한 224×224는 모양이 달라질 수 있음을 설명합니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   핵심은 BGR로 바꿨다가 다시 RGB로 바꾸는 것이 아닙니다. 
#   OpenCV가 처음부터 이미지를 BGR 순서로 읽어오기 때문에, 필요한 목적에 따라 GRAY 또는 RGB로 변환하는 것입니다.
# <<< 해설 B02
small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
cv2.imwrite("p02_gray.png", gray)
cv2.imwrite("p02_small.png", small)
# RGB 배열은 화면 표시용입니다. imwrite에 그대로 넣지 않습니다.
print("Gray shape:", gray.shape)
print("RGB 빨강:", rgb[30, 30].tolist())
print("Resize shape:", small.shape)
