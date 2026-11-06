"""실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent
history = pd.read_csv(BASE / "data/metrics_example.csv")
# >>> B06
# 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
# 해설: 이 실습의 선택 규칙이며 Ultralytics 내부 best fitness와 같다고 가정하지 않습니다.
best = history.loc[history["map50_95"].idxmax()]
# <<< B06
print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")
print("선택 epoch:", int(best["epoch"]))
print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])
print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")

"""
20261106,   TASK01

최종 결론: 
mAP50 대비 mAP50-95가 낮아, 객체를 탐지하더라도 Bounding Box의 위치 정밀도가 충분하지 않을 가능성이 있다. 
따라서 향후 객체 위치 예측 성능과 클래스별 탐지 성능을 추가 분석할 필요가 있다.
"""