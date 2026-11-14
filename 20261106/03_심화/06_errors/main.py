"""실습 06. 오류 분류 원리. 손계산 fixture를 사용하며 실제 YOLO 오류가 아닙니다."""
from pathlib import Path
import sys
import pandas as pd
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/"provided"))
from geometry import iou_matrix
from metrics import match_predictions

out=BASE/"outputs";out.mkdir(exist_ok=True)
p=pd.read_csv(BASE.parent/"data/boxes_fixture.csv").to_dict("records")
g=pd.read_csv(BASE.parent/"data/boxes_gt.csv").to_dict("records")

# TODO C06: IoU .5 일대일 매칭을 호출하세요. 결과의 gt_index가 None이면 FP입니다.
# records: 각 예측이 어떤 정답과 연결됐는지 기록한 목록. 예측 하나당 딕셔너리 하나가 만들어짐
records,counts=match_predictions(predictions=p, ground_truth=g, iou_threshold=.5)

keys=["x1","y1","x2","y2"]
rows=[]
for r in records:
    q=p[r["prediction_index"]]
    # 예측박스와 정답박스 간 IoU 계산
    overlap=iou_matrix([[q[k] for k in keys]],[[t[k] for k in keys] for t in g])[0]

    # 클래스 일치 확인
    j=int(overlap.argmax());same=q["class_id"]==g[j]["class_id"]

    # 예측 하나의 결과를 보고, TP인지 또는 어떤 종류의 오류인지 reason에 기록
    # 같은 정답을 예측 두 개가 겹쳐 탐지했다면, 먼저 매칭된 예측은 TP, 남은 예측은 Duplicate로 분류돼요.
    reason="TP" if r["tp"] else ("Duplicate" if same and overlap[j]>=.5 else "Wrong_class" if not same and overlap[j]>=.5 else "Loose_box" if same and overlap[j]>=.1 else "Background_FP")
    rows.append(dict(prediction_index=r["prediction_index"],reason=reason,best_iou=float(overlap[j])))
pd.DataFrame(rows).to_csv(out/"fixture_errors.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False));print(counts)
print("Miss는 매칭되지 않은 정답에서 따로 집계합니다.")

# 실습목적: 일대일 매칭에 따른 탐지 결과 집계와 오류 분류 원리 파악
# 결론: 예측 4건과 정답 3건을 비교하여 TP 2건, FN 1건, FP 2건을 확인하였다.
# FP 2건은 중복 탐지(Duplicate) 1건과 정답 물체와 겹치지 않는 오탐(Background_FP) 1건이었다.