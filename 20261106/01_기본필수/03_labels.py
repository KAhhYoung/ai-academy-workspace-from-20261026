"""실습 03. 정상 이미지 6장과 YOLO TXT 라벨을 읽습니다."""
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
names = ["square", "circle", "triangle"]

for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:
    # 한글경로 오류 발생
    #image = cv2.imread(str(image_path))
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    lines=[]
    # TODO B04: UTF-8 라벨 파일의 문자열을 읽고 줄 목록으로 나누세요.
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())

    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))
    for line in lines:
        cls, cx, cy, width, height = map(float, line.split())
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)
