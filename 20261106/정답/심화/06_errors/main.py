
"""
실습 06: IoU 기반 객체 탐지 오류 유형 분석

[실습 목적]
예측 Bounding Box와 실제 정답(Ground Truth)을 비교하여
TP, FP, FN을 집계하고 오탐의 원인을 분류합니다.

[주요 동작]
1. 예측 데이터와 정답 데이터 불러오기
2. IoU 0.5 기준으로 예측과 정답을 일대일 매칭
3. Duplicate, Wrong_class, Loose_box, Background_FP 분류
4. 오류 분석 결과를 CSV로 저장

※ 실제 YOLO 추론 결과가 아닌 실습용 가상 데이터(Fixture)입니다.
"""

from pathlib import Path
import sys
import pandas as pd

# ============================================================
# 1. 경로 및 제공 모듈 설정
# ============================================================

BASE = Path(__file__).resolve().parent

# 제공된 IoU 계산 및 매칭 함수가 있는 폴더 추가
sys.path.insert(0, str(BASE.parent / "provided"))

from geometry import iou_matrix
from metrics import match_predictions

# 결과 저장 폴더 생성
out = BASE / "outputs"
out.mkdir(exist_ok=True)


# ============================================================
# 2. 예측 데이터 및 실제 정답 로딩
# ============================================================

# 예측 Bounding Box 목록 (Prediction)
p = pd.read_csv(
    BASE.parent / "data/boxes_fixture.csv"
).to_dict("records")

# 실제 정답 Bounding Box 목록 (Ground Truth)
g = pd.read_csv(
    BASE.parent / "data/boxes_gt.csv"
).to_dict("records")


# ============================================================
# 3. IoU 기반 예측·정답 매칭
# ============================================================

# IoU >= 0.5 기준으로 예측과 정답을 일대일 매칭
# 하나의 정답 객체는 한 번만 TP로 인정
#
# records: 예측별 TP 여부와 매칭 정보
# counts : TP, FP, FN 집계 결과
records, counts = match_predictions(p, g, 0.5)


# ============================================================
# 4. 객체 탐지 오류 유형 분석
# ============================================================

# Bounding Box 좌표 컬럼
keys = ["x1", "y1", "x2", "y2"]

rows = []

for r in records:

    # 현재 분석할 예측 Bounding Box
    q = p[r["prediction_index"]]

    # 현재 예측 박스와 모든 정답 박스의 IoU 계산
    overlap = iou_matrix(
        [[q[k] for k in keys]],
        [[t[k] for k in keys] for t in g]
    )[0]

    # IoU가 가장 높은 정답 객체 선택
    j = int(overlap.argmax())

    # 예측 클래스와 정답 클래스가 동일한지 확인
    same = q["class_id"] == g[j]["class_id"]

    # --------------------------------------------------------
    # 5. 오류 원인 분류
    # --------------------------------------------------------

    # TP            : 올바르게 매칭된 예측
    # Duplicate     : 같은 객체를 중복 탐지
    # Wrong_class   : 위치는 겹치지만 클래스가 다름
    # Loose_box     : 클래스는 같지만 박스 위치 정확도가 부족
    # Background_FP : 정답 객체와 충분히 겹치지 않는 오탐
    reason = (
        "TP" if r["tp"]
        else "Duplicate" if same and overlap[j] >= 0.5
        else "Wrong_class" if not same and overlap[j] >= 0.5
        else "Loose_box" if same and overlap[j] >= 0.1
        else "Background_FP"
    )

    # 예측 번호, 오류 유형, 최대 IoU 기록
    rows.append(
        dict(
            prediction_index=r["prediction_index"],
            reason=reason,
            best_iou=float(overlap[j])
        )
    )


# ============================================================
# 6. 분석 결과 저장 및 출력
# ============================================================

# 오류 분석 결과를 CSV 파일로 저장
df = pd.DataFrame(rows)

df.to_csv(
    out / "fixture_errors.csv",
    index=False
)

# 예측별 오류 유형 출력
print(df.to_string(index=False))

# 전체 TP / FP / FN 출력
print(counts)

# FN은 예측별 오류 목록이 아니라 미매칭 정답에서 집계
print("Miss는 매칭되지 않은 정답에서 따로 집계합니다.")
