"""실습 03. 정상 이미지 6장과 YOLO TXT 라벨을 읽습니다.

개요: 이번 스크립트는 이미지와 라벨 파일이 대응하는지, 좌표 형식이 어떻게 구성되는지 확인하는 용도입니다
"""
from pathlib import Path
import cv2
BASE = Path(__file__).resolve().parent
names = ["square", "circle", "triangle"]
for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:
    image = cv2.imread(str(image_path))
    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    # >>> B04
    # 요구: UTF-8 라벨 파일의 문자열을 읽고 줄 목록으로 나누세요.
    # 해설: 이 입력은 모두 3개의 정상 객체를 포함합니다.
    lines = label_path.read_text(encoding="utf-8").splitlines()
    # <<< B04
    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))
    for line in lines:
        cls, cx, cy, width, height = map(float, line.split())
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)

"""
basic_train_000.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.796875 0.33984375 0.19375 0.2421875
  circle normalized: 0.1875 0.25 0.1625 0.203125
  triangle normalized: 0.48125 0.640625 0.2125 0.265625
basic_train_001.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.23125 0.234375 0.225 0.28125
  circle normalized: 0.50625 0.6875 0.2 0.25
  triangle normalized: 0.834375 0.27734375 0.23125 0.2890625
basic_train_002.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.165625 0.32421875 0.21875 0.2734375
  circle normalized: 0.79375 0.3046875 0.2 0.25
  triangle normalized: 0.55 0.65625 0.1875 0.234375
basic_train_003.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.83125 0.25 0.175 0.21875
  circle normalized: 0.178125 0.29296875 0.19375 0.2421875
  triangle normalized: 0.534375 0.64453125 0.18125 0.2265625
basic_train_004.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.853125 0.26953125 0.18125 0.2265625
  circle normalized: 0.453125 0.76171875 0.23125 0.2890625
  triangle normalized: 0.215625 0.33984375 0.23125 0.2890625
basic_train_005.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.1625 0.328125 0.1875 0.234375
  circle normalized: 0.846875 0.28515625 0.16875 0.2109375
  triangle normalized: 0.54375 0.734375 0.2 0.25
"""