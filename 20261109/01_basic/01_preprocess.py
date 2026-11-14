"""실습 01: 정상 문서 3장을 Gray와 Otsu로 변환합니다. 원본을 덮어쓰지 않습니다."""
from pathlib import Path
import cv2

base = Path(__file__).resolve().parent
for name in ['doc01', 'doc02', 'doc03']:
    image = cv2.imread(str(base / f'{name}.png'))
# TODO B01: gray = image[:, :, 0] * 0  → cv2.cvtColor() 함수 사용
#           threshold, binary = 0, gray → cv2.threshold() 함수 사용
#           출력은 채널이 없는 (높이, 너비) 배열입니다.
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    # Otsu 사용 시 thresh(기준값) 무시됨
    threshold, binary = cv2.threshold(src=gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(str(base / f'p01_{name}_gray.png'), gray)
    cv2.imwrite(str(base / f'p01_{name}_binary.png'), binary)
    print(name, 'shape:', gray.shape, 'threshold:', threshold)
