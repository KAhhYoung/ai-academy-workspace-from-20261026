"""실습 05. 실제 준비 모델을 valid에 평가합니다. test는 전달하지 않습니다."""
from pathlib import Path
import json,yaml,torch
import pandas as pd
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent

if __name__=="__main__":
    torch.set_num_threads(1)
    out=BASE/"outputs";out.mkdir(exist_ok=True)
    config={
        "train":str(BASE/"data/images/train"),
        "val":str(BASE/"data/images/val"),
        "names":["square","circle","triangle"]
    }
    path=out/"dataset_active.yaml"
    path.write_text(yaml.safe_dump(config),encoding="utf-8")

    #model=YOLO(str(BASE/"weights"/"demo.pt"))
    model=YOLO(str(BASE/"outputs"/"smoke2"/"weights"/"best.pt"))
    # TODO A05: model.val에 data=path, split='val'을 전달하세요.
    # 사전 학습된 가중치를 사용하더라도 하이퍼파라미터를 변경하고 모델 개발 과정에 사용하는 거라면 predict 아닌 val (모델 가중치 변경 없음)
    # [Q] 라벨 경로를 직접 전달하지 않았는데 어떻게 성능평가를 할까?
    # [A] yaml 에 전달한 "val": str(BASE / "data/images/val") 를 보고 YOLO가 라벨 경로를 찾음
    #     일반적인 YOLO 데이터셋 규칙에 따라 경로의 images를 labels로 바꾸고, 같은 이름의 .txt 파일을 찾음
    metrics=model.val(
        data=str(path),
        split='val',
        project=str(out),
        name="validation",
        imgsz=160,
        batch=4,
        workers=0,
        device="cpu",
        plots=True,
        exist_ok=True
    )

    summary={"map50":float(metrics.box.map50),"map50_95":float(metrics.box.map),"split":"val"}
    (out/"validation"/"validation_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    rows=[]
    for i,cls in enumerate(metrics.box.ap_class_index):
        pr,rc,ap50,ap=metrics.box.class_result(i)
        rows.append([int(cls),model.names[int(cls)],float(pr),float(rc),float(ap50),float(ap)])
    pd.DataFrame(rows,columns=["class_id","name","precision","recall","ap50","ap50_95"]).to_csv(out/"validation"/"class_metrics.csv",index=False)
    print(summary)


# 결과:
# 검증 과정과 정답 라벨 로딩은 정상적으로 수행되었으나, 현재 평가한 demo.pt는 검증 물체 30개를 모두 미탐지하였다.
# 모든 클래스의 Precision, Recall 및 AP가 0으로 나타났으며, 모델의 학습 상태와 평가 가중치의 적절성을 추가 확인해야 한다.
