import cv2
import numpy as np

def apply_filter(gray, name):
    """한 장의 Gray에 선택 필터를 적용합니다. 파일 입출력은 하지 않습니다."""
    if name == "gaussian":
        return cv2.GaussianBlur(gray,(5,5),0)
    if name == "sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
        return cv2.filter2D(gray,-1,kernel)
    # >>> 해설 H03
    # Bilateral은 공간 거리와 밝기 차이를 함께 반영합니다. 항상 다른 필터보다 우수하거나 빠르다는 뜻은 아닙니다.
    return cv2.bilateralFilter(gray, 7, 50, 50)
    # <<< 해설 H03
