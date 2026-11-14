"""실습 02. 고전 NMS 원리 실험. YOLO26 기본 추론에 NMS가 있다고 가정하지 않습니다."""
from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou, batched_nms
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
p=pd.read_csv(BASE/"data/boxes_fixture.csv");g=pd.read_csv(BASE/"data/boxes_gt.csv")
boxes=torch.tensor(p[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
scores=torch.tensor(p.confidence.to_numpy(),dtype=torch.float32)
classes=torch.tensor(p.class_id.to_numpy(),dtype=torch.int64)
gb=torch.tensor(g[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
rows=[]
for conf,nms_iou in [(0.25,0.5),(0.5,0.5),(0.25,0.95)]:
    ids=torch.where(scores>=conf)[0]
    # TODO A02: 같은 클래스끼리만 중복을 제거하는 batched_nms를 적용하세요.
    # 반환 index는 ids로 줄인 배열 기준이므로 다시 원본 index로 바꿉니다.

    # ------------------------------------------
    # NMS 수행(예측박스 간 IoU 계산) <-- 임계값 변화
    # ------------------------------------------
    keep = ids[
        batched_nms(
            boxes[ids],
            scores[ids],
            classes[ids],
            nms_iou
        )
    ]

    # ------------------------------------------
    # 예측박스와 정답박스 간 IoU 계산 <-- 임계값 고정
    # 클래스 일치
    # TP 1:1 매칭
    # TP, FP, FN 결정
    # Precision, Recall 계산
    # ------------------------------------------
    used=set();tp=0
    for i in keep.tolist():
        ovs=box_iou(boxes[[i]],gb)[0]
        eligible=[j for j in range(len(g)) if int(classes[i])==int(g.iloc[j].class_id) and j not in used]
        #for j in sorted(eligible,key=lambda j:-float(ovs[j])):
        for j in sorted(eligible, key=lambda j: float(ovs[j]), reverse=True):    
            if ovs[j]>=0.5:used.add(j);tp+=1;break
    n=len(keep);fp=n-tp;fn=len(g)-tp
    rows.append(dict(conf=conf,nms_iou=nms_iou,kept=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0,recall=tp/len(g)))
r=pd.DataFrame(rows);r.to_csv(out/"nms_comparison.csv",index=False);print(r.round(4).to_string(index=False))

# 작성자: 김아영
# 실습목적: 중복예측을 거르고 conf,nms_iou를 적용하여 tp,fp,fn,precision,recall 을 확인

# 실습결과:
# |  No. | conf | nms_iou | kept | tp | fp | fn | precision | recall |
# |------|------|---------|------|----|----|----|-----------|--------|
# |   1  | 0.25 | 0.50    | 3    | 2  | 1  | 1  | 0.6667    | 0.6667 |
# |   2  | 0.50 | 0.50    | 2    | 2  | 0  | 1  | 1.0000    | 0.6667 |
# |   3  | 0.25 | 0.95    | 4    | 2  | 2  | 1  | 0.5000    | 0.6667 |

# 해석:
# 1번 설정에서는, 정답객체 3개 중 2개를 찾아냈고 예측한 3개 중 1개는 오탐하였다.
# 신뢰도임계치만 0.5로 상향한 2번 설정에서는, 오탐이 1개에서 0개로 줄고, 올바른 검출 수와 미검출 수는 그대로여서 precision이 100%로 올라갔다.
# nms_iou 임계값만 0.95로 상향한 3번 설정에서는, 1번 설정 대비 오탐이 1개에서 2개로 늘었고 올바른 검출 수와 미검출 수는 그대로여서 precision이 줄었다.

# 결론:
# conf=0.5, nms_iou=0.5 인 2번 설정이 이 데이터에서는 더 좋은 선택이다.

# 보완사항:
# 이 설정이 다른 이미지에서도 좋은지는 추가 평가가 필요하다.
# 여러 이미지에 해당하는 예측 박스(boxes_fixture)와 정답 박스(boxes_gt)가 필요하다.
