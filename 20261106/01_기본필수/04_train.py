"""실습 04. 무작위 초기화 YOLO26의 1 epoch 학습 연결 점검. 성능 실험이 아닙니다."""
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
BASE = Path(__file__).resolve().parent
# Windows의 데이터 로딩 worker가 학습 코드를 중복 실행하지 않도록 합니다.
if __name__ == "__main__":
    torch.set_num_threads(1)
    config = {
        "train": str(BASE/"data/images/train"),
        "val": str(BASE/"data/images/val"),
        "names": ["square", "circle", "triangle"]
    }
    config_path = BASE / "p04_dataset.yaml"

    # 키로 정렬하므로 순서 달라질 수 있음
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = YOLO("yolo26n.pt")  # 구조만 생성합니다. COCO 사전학습 .pt와 다릅니다.
    # TODO B05: config_path로 1 epoch, imgsz=160, batch=4, CPU 학습을 호출하세요.
    # 나머지 옵션은 제공한 그대로 유지하세요. test 경로는 전달하지 않습니다.
    model.train(
        data=str(config_path),
        epochs=1,
        imgsz=160,
        batch=4,
        device="cpu"
    )

    print("task:", model.task, "names:", model.names, "device:", model.device)
    print("학습 결과:", BASE/"p04_runs/smoke")
