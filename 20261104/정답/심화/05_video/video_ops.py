import cv2

def process_frame(frame):
    """컬러 Frame을 160×120 크기의 Edge 컬러 Frame으로 변환합니다."""

    # ① Resize: 이미지 크기를 160×120으로 축소
    # INTER_AREA: 이미지 축소 시 픽셀 영역을 고려하여 품질 저하를 줄임
    # 영상 처리 연산량을 줄이고 처리 속도를 높이기 위한 과정
    small = cv2.resize(
        frame, (160, 120), interpolation=cv2.INTER_AREA
    )

    # ② Grayscale: BGR 컬러 이미지를 흑백(1채널) 이미지로 변환
    # Canny는 밝기 변화로 경계를 검출하므로 색상 정보가 필요하지 않음
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # ③ Gaussian Blur: 3×3 가우시안 필터로 이미지의 노이즈를 완화
    # 작은 노이즈가 불필요한 Edge로 검출되는 것을 줄이기 위한 전처리
    # (3,3): 커널 크기, 0: Sigma 값을 OpenCV가 자동 계산
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # ④ Canny Edge Detection: 밝기가 급격하게 변하는 경계 검출
    # 80 : 하한 임계값 (약한 경계 판단 기준)
    # 160: 상한 임계값 (강한 경계 판단 기준)
    # 결과는 경계가 흰색(255), 나머지는 검정(0)인 1채널 이미지
    edge = cv2.Canny(blurred, 80, 160)

    # ⑤ Grayscale → BGR 변환
    # Edge 결과는 1채널이므로 3채널 BGR 형식으로 변환
    # 컬러 영상과 동일한 채널 형식으로 출력하기 위한 과정
    # 단, 원본 컬러가 복원되는 것은 아님
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
