
"""
실습 04: YOLO26n 학습 및 실행 시간 측정

[실습 목적]
YOLO 객체 탐지 모델을 학습하고, 학습 완료까지 소요된 시간을 측정합니다.

[주요 동작]
1. 데이터셋 YAML 설정 생성
2. YOLO26n 모델 생성
3. 2 Epoch Smoke Test 학습
4. 전체 학습 소요 시간 측정
5. API와 동일한 설정의 CLI 명령 저장
"""

from pathlib import Path
import time
import yaml
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":

    torch.set_num_threads(1)

    # 1. 결과 저장 폴더 생성
    out = BASE / "outputs"
    out.mkdir(exist_ok=True)

    # 2. 학습 데이터셋 경로 설정
    config = {
        "train": str(BASE / "data/images/train"),
        "val": str(BASE / "data/images/val"),
        "names": ["square", "circle", "triangle"]
    }

    # YOLO 학습용 YAML 생성
    path = out / "dataset_active.yaml"
    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8"
    )

    # 3. YOLO26n 모델 생성
    model = YOLO("yolo26n.yaml")

    # 4. 학습 시간 측정 시작
    start_time = time.perf_counter()

    # 5. YOLO 학습 실행
    model.train(
        data=str(path),
        epochs=2,
        imgsz=160,
        batch=4,
        workers=0,
        device="cpu",
        amp=False,
        pretrained=False,
        optimizer="SGD",
        lr0=0.01,
        seed=42,
        plots=False,
        project=str(out),
        name="smoke",
        exist_ok=True
    )

    # 6. 학습 완료 시간 측정
    end_time = time.perf_counter()

    # 총 소요 시간 (초)
    elapsed = end_time - start_time

    # 분 / 초 단위로 변환
    minutes, seconds = divmod(elapsed, 60)

    # 7. API와 동일한 설정의 CLI 명령 저장
    command = (
        f'yolo detect train model=yolo26n.yaml '
        f'data="{path}" epochs=2 imgsz=160 '
        f'batch=4 workers=0 device=cpu amp=False '
        f'pretrained=False optimizer=SGD lr0=0.01 '
        f'seed=42 plots=False project="{out}" '
        f'name=cli_smoke exist_ok=True'
    )

    (out / "same_settings_cli.txt").write_text(
        command,
        encoding="utf-8"
    )

    # 8. 최종 결과 출력
    print("\n" + "=" * 55)
    print("          YOLO26n 학습 결과")
    print("=" * 55)
    print(f"학습 장치       : CPU")
    print(f"학습 Epoch      : 2")
    print(f"Batch Size      : 4")
    print(f"Image Size      : 160 × 160")
    print(f"총 학습 소요 시간: {elapsed:.2f}초")
    print(f"시간 환산       : {int(minutes)}분 {seconds:.2f}초")
    print(f"가중치 저장 위치: {out / 'smoke/weights'}")
    print("=" * 55)

    # 9. 학습 시간 별도 파일 저장
    (out / "training_time.txt").write_text(
        f"total_seconds={elapsed:.4f}\n"
        f"epochs=2\n"
        f"device=cpu\n",
        encoding="utf-8"
    )
