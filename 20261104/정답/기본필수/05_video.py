from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

cap = cv2.VideoCapture("sample.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("p05_gray.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
print("열림(입력/출력):", cap.isOpened(), writer.isOpened())
print("입력 FPS/크기:", fps, (width, height))
count = 0
while cap.isOpened() and writer.isOpened():
    ok, frame = cap.read()
    if not ok:  # 영상 끝에 도달하면 반복을 끝냅니다. 예외처리 과제가 아닙니다.
        break
    # >>> 해설 B05
    # read는 (성공 여부, 프레임)을 반환합니다. 매번 하나의 프레임만 처리합니다. MJPG는 손실 코덱이므로 PNG처럼 픽셀 완전 일치를 요구하지 않습니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # <<< 해설 B05
    # 컬러 Writer에 맞춰 Gray를 3채널로 돌립니다. 색이 생기는 것은 아닙니다.
    writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    count += 1
cap.release()
writer.release()
check = cv2.VideoCapture("p05_gray.avi")
print("처리/재읽기 Frame:", count, int(check.get(cv2.CAP_PROP_FRAME_COUNT)))
print("재읽기 FPS:", check.get(cv2.CAP_PROP_FPS))
check.release()
