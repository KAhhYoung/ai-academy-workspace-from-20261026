import cv2
import numpy as np

def prepare_image(image,size=(160,120)):
    """색공간·Resize·필터·형태학 처리의 중간 결과를 딕셔너리로 반환합니다."""
    # >>> TODO H07: Gray와 size=(너비,높이)로 축소한 small을 만드세요. 원본 image를 직접 수정하지 않습니다.
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,size)

    blur=cv2.GaussianBlur(src=small,ksize=(3,3),sigmaX=0)
    _,binary=cv2.threshold(src=blur,thresh=127,maxval=255,type=cv2.THRESH_BINARY)

    # 학습하는 초기 가중치가 아니라, 연산에 포함할 위치를 지정하는 마스크입니다. (모든 값이 1)
    # - 1: 해당 위치를 연산에 포함
    # - 0: 해당 위치를 제외
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # cv2.MORPH_OPEN: 침식 후 팽창
    # cv2.MORPH_CLOSE: 팽창 후 침식
    # 침식: 픽셀 주변(커널범위)이 전부 흰색일 때만 중심을 흰색으로 남깁니다.
    # 팽창: 픽셀 주변(커널범위)에 흰색이 하나라도 있으면, 현재 위치의 결과를 흰색으로 만드는 연산
    opened=cv2.morphologyEx(src=binary,op=cv2.MORPH_OPEN,kernel=kernel)
    edge=cv2.Canny(image=blur,threshold1=80,threshold2=160)
    return {"gray":gray,"small":small,"blur":blur,"binary":binary,"opened":opened,"edge":edge}

def augment_image(image):
    """종합 실습에서는 설정을 고정해 재실행 비교를 쉽게 합니다."""
    flipped=cv2.flip(image,1)
    return np.clip(flipped.astype(np.float32)+15,0,255).astype(np.uint8)
