
from pathlib import Path
import os
import cv2
import numpy as np

# ============================================================
# 1. 실행 환경 및 경로 설정
# ============================================================

# Python 파일이 위치한 폴더를 현재 작업 디렉터리로 설정
# 어느 위치에서 실행하더라도 상대 경로가 동일하게 동작하도록 함
os.chdir(Path(__file__).resolve().parent)

import csv
import json
from time import perf_counter

# 처리 결과를 저장할 outputs 폴더 생성
# exist_ok=True: 폴더가 이미 있어도 오류가 발생하지 않음
Path("outputs").mkdir(exist_ok=True)


# ============================================================
# 2. 이미지 불러오기 및 이진화(Binary Thresholding)
# ============================================================

# 이미지를 Grayscale(흑백, 1채널)로 읽기
# 두 번째 인수 0은 cv2.IMREAD_GRAYSCALE과 동일
# 픽셀 값의 범위: 0(검정) ~ 255(흰색)
gray = cv2.imread("../data/objects.png", 0)

# Threshold(임계값)를 기준으로 이미지를 흑백으로 구분
# 픽셀 값 > 127  → 255(흰색, 전경)
# 픽셀 값 <= 127 → 0(검정, 배경)
#
# 반환값:
# 첫 번째: 실제 적용된 임계값
# 두 번째: 이진화된 이미지
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# 결과를 저장할 리스트
rows = []


# ============================================================
# 3. 커널(Kernel)의 모양별 반복 실험
# ============================================================

# 형태학적 연산에 사용할 커널 모양 비교
#
# MORPH_RECT    : 직사각형 커널
# MORPH_ELLIPSE : 타원형 커널
#
# 커널은 이미지 위를 이동하며 주변 픽셀을 검사하는 작은 행렬
for shape_name, shape in [
    ("rect", cv2.MORPH_RECT),
    ("ellipse", cv2.MORPH_ELLIPSE)
]:

    # --------------------------------------------------------
    # 4. 커널 크기 설정
    # --------------------------------------------------------

    # 3×3 또는 5×5 크기의 커널을 사용
    # 일반적으로 커널이 클수록 더 넓은 영역에 영향을 줌
    for size in [3, 5]:

        # ----------------------------------------------------
        # 5. 연산 반복 횟수 설정
        # ----------------------------------------------------

        # iterations=1: 연산 1회
        # iterations=2: 연산 2회
        #
        # 반복 횟수가 증가하면 형태학적 연산의 효과가
        # 더 강하게 나타날 수 있음
        for iterations in [1, 2]:

            # ------------------------------------------------
            # 6. 구조 요소(Structuring Element) 생성
            # ------------------------------------------------

            # 선택한 모양과 크기에 맞는 커널 생성
            #
            # 예: MORPH_RECT, (3,3)
            #
            # [[1, 1, 1],
            #  [1, 1, 1],
            #  [1, 1, 1]]
            #
            # 커널의 모양과 크기에 따라
            # 제거되거나 유지되는 객체의 형태가 달라짐
            kernel = cv2.getStructuringElement(
                shape, (size, size)
            )

            # ------------------------------------------------
            # 7. 형태학적 열림 연산(Opening)
            # ------------------------------------------------

            # MORPH_OPEN = 침식(Erosion) → 팽창(Dilation)
            #
            # 침식:
            # - 흰색 영역의 경계를 줄임
            # - 작은 흰색 노이즈를 제거할 수 있음
            #
            # 팽창:
            # - 남아 있는 흰색 영역을 다시 확장
            # - 침식 후 남은 객체의 크기를 일정 부분 복원
            #
            # 주요 목적:
            # - 작은 흰색 노이즈 제거
            # - 가느다란 연결부 제거
            # - 객체 외곽의 작은 돌출부 제거
            #
            # iterations=2이면
            # 침식 2회 후 팽창 2회를 수행
            result = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel,
                iterations=iterations
            )

            # ------------------------------------------------
            # 8. 연결요소 분석(Connected Components)
            # ------------------------------------------------

            # 이진 이미지에서 서로 연결된 흰색 영역을 검출
            #
            # connectivity=8:
            # 상하좌우 + 대각선 방향까지 연결된 것으로 판단
            #
            # 반환값:
            # count     : 배경을 포함한 전체 Label 개수
            # labels    : 각 픽셀에 부여된 Label 번호
            # stats     : 각 객체의 위치, 너비, 높이, 면적
            # centroids : 각 객체의 중심 좌표
            #
            # Label 0은 배경이므로 실제 객체 개수는 count-1
            count, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    result,
                    connectivity=8
                )
            )

            # ------------------------------------------------
            # 9. 결과 파일 이름 생성
            # ------------------------------------------------

            # 커널 모양, 크기, 반복 횟수를 파일 이름에 기록
            #
            # 예: rect_3_1.png
            #     └ 직사각형, 3×3, 반복 1회
            #
            # 예: ellipse_5_2.png
            #     └ 타원형, 5×5, 반복 2회
            name = f"{shape_name}_{size}_{iterations}"

            # ------------------------------------------------
            # 10. 처리된 이미지 저장
            # ------------------------------------------------

            # 형태학적 연산 결과를 PNG 파일로 저장
            # 원본 이미지는 수정하지 않음
            cv2.imwrite(
                f"outputs/{name}.png",
                result
            )

            # ------------------------------------------------
            # 11. 분석 결과 수집
            # ------------------------------------------------

            # np.count_nonzero(result):
            # 값이 0이 아닌 픽셀의 개수 계산
            # 현재 이진 이미지에서는 흰색 전경 픽셀 수를 의미
            #
            # count-1:
            # 배경(Label 0)을 제외한 연결요소 개수
            #
            # 주의:
            # 연결요소 개수 = 항상 실제 물체 개수는 아님
            # 노이즈도 별도 객체로 검출될 수 있음
            rows.append([
                name,
                int(np.count_nonzero(result)),
                count - 1
            ])


# ============================================================
# 12. 실험 결과 CSV 저장
# ============================================================

# 총 실험 조합:
# 커널 모양 2개 × 크기 2개 × 반복 횟수 2개 = 8개
#
# 저장 항목:
# setting: 적용한 커널 모양/크기/반복 횟수
# foreground_pixels: 처리 후 남은 흰색 픽셀 수
# components_without_background: 배경 제외 연결요소 수
with open(
    "outputs/morphology.csv",
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    w = csv.writer(f)

    # CSV 헤더 작성
    w.writerow([
        "setting",
        "foreground_pixels",
        "components_without_background"
    ])

    # 8가지 실험 결과 기록
    w.writerows(rows)


# ============================================================
# 13. 최종 결과 출력
# ============================================================

# 커널 설정에 따른 전경 픽셀 수와 객체 수 비교
import pandas as pd

df = pd.DataFrame(
    rows,
    columns=["Kernel Setting", "Foreground Pixels", "Components"]
)

print("\n[OpenCV Morphology Opening Results]\n")
print(df.to_markdown(index=False, tablefmt="grid"))
