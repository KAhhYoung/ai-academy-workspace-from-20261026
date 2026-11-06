"""실습 01. 정답 박스 3개를 그립니다. 원본 scene.png는 바꾸지 않습니다."""
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
BASE = Path(__file__).resolve().parent

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# 한글경로 오류 문제 발생
#image = cv2.imread(str(BASE / "data/scene.png"))

data = np.fromfile(str(BASE / "data/scene.png"), dtype=np.uint8)
image = cv2.imdecode(data, cv2.IMREAD_COLOR)

boxes = pd.read_csv(BASE / "data/boxes_gt.csv")
result = image.copy()  # 그릴 배열을 복사하여 원본을 보존합니다.
for row in boxes.itertuples():
    # >>> B01
    # 요구: result에 (x1,y1)~(x2,y2) 박스를 두께 2로 그리세요.
    # cv2.rectangle의 이미지·두 점·색은 필수, thickness는 선택입니다.
    cv2.rectangle(img=result, pt1=(row.x1, row.y1), pt2=(row.x2, row.y2), color=RED, thickness=2)
    print("class:", row.class_id, "xyxy:", row.x1, row.y1, row.x2, row.y2, "GT confidence: N/A")
cv2.imwrite(str(BASE / "p01_boxes.png"), result)
print("shape:", image.shape, "저장: p01_boxes.png")
