"""실습 04. API 학습과 같은 설정의 CLI 명령을 기록합니다."""
from pathlib import Path
import yaml,torch
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent

if __name__=="__main__":
    torch.set_num_threads(1)
    out=BASE/"outputs";out.mkdir(exist_ok=True)

    config={"train":str(BASE/"data/images/train"),
            "val":str(BASE/"data/images/val"),
            "names":["square","circle","triangle"]
    }
    path=out/"dataset_active.yaml"
    path.write_text(yaml.safe_dump(config),encoding="utf-8")

    # 모델 구조로 새로 생성 → 처음부터 학습
    #model = YOLO("yolo26n.yaml")

    # 사전학습 가중치에서 시작해 도형 데이터로 추가 학습
    #model = YOLO("yolo26n.pt")

    # 사전 학습된 가중치 로드 → 전이학습(여기서는 fine-tuning)
    model = YOLO(str(BASE/"weights"/"demo.pt"))
    # TODO A04: epoch smoke 학습을 실행하세요. seed·imgsz·batch·optimizer 조건을 고정합니다.
    # smoke test: 학습 과정이 정상적으로 돌아가는지 짧게 확인하는 실행
    #             모델 성능을 충분히 높이는 것이 아니라, 학습 준비와 실행에 문제가 없는지 확인하는 테스트
    # smoke test 로 확인할 사항
    #   -데이터와 라벨을 정상적으로 읽는지
    #   -모델 학습이 오류 없이 진행되는지
    #   -결과와 가중치가 저장되는지

    # project: 결과를 저장할 상위 폴더
    # name="smoke"는 실험 결과 폴더 이름
    # project와 name으로 결과 폴더를 정하면, Ultralytics가 그 안에 weights 폴더를 자동으로 생성

    # 실습목적
    # 기존 학습된 가중치를 새 데이터로 학습 → 미세조정(fine-tuning)
    # 기본적으로는 학습 가능한 전체 계층의 가중치를 갱신. 일부 계층을 고정하고 학습하려면 freeze를 지정할 수 있음
    # 다만 같은 데이터로 기존 학습을 계속하는 경우는 보통 학습 재개(resume training) 라고 부름
    # .pt를 불러오는 것 자체보다, 기존에 배운 지식을 새로운 데이터나 작업에 활용하는지가 전이학습의 핵심임
    model.train(
        data=str(path),
        epochs=250,
        patience=0,
        seed=42,
        deterministic=True,
        imgsz=160,
        batch=4,
        device="cpu",
        optimizer="SGD",
        lr0=0.01,
        project=str(out),
        name="smoke",
        resume=False,
    )

    trainer = model.trainer
    stopper = trainer.stopper

    print("최고 성능 epoch:", stopper.best_epoch)
    print("최고 성능 점수:", stopper.best_fitness)
    print("마지막 epoch:", trainer.epoch + 1)

    # 조기 종료 기준에 도달했는지 확인
    no_improvement = (trainer.epoch + 1) - stopper.best_epoch
    print("성능 개선 없는 epoch 수:", no_improvement)
    print("조기 종료 기준 도달:", no_improvement >= stopper.patience)

    '''참고사항
    기존 가중치로 새 학습을 시작하는 것과 중단된 학습을 재개하는 것은 다름
    중단 지점의 epoch·옵티마이저 상태까지 복원하려면, 재개 가능한 last.pt를 불러와 다음처럼 사용
    model = YOLO(str(out / "smoke" / "weights" / "last.pt"))
    model.train(resume=True)
    '''

    # imgsz: 입력 이미지 크기 기준
    command=f'yolo detect train model=demo.pt data="{path}" epochs=2 imgsz=160 batch=4 workers=0 device=cpu amp=False pretrained=False optimizer=SGD lr0=0.01 seed=42 plots=False project="{out}" name=cli_smoke exist_ok=True'
    (out/"same_settings_cli.txt").write_text(command,encoding="utf-8")
    print("best/last:",out/"smoke/weights", "CLI 비교는 선택 실행입니다.")

# 스모크 테스트 결론
# 스모크 테스트로 2 epoch 동안 학습과 검증이 실행되어 학습 과정이 정상적으로 실행되는지는 확인했지만, 탐지 성능을 확보했다고 보기는 어렵다.

# ----------------------------------------------------------------
# 학습 진행 결과
# 최대 epoch별 학습 결과 비교 (patience=0, deterministic=True 로 고정)
# 최대 epoch | 최고 성능 epoch | 최고 mAP50-95
# -----------|----------------|--------------
# 100        | 97             | 0.78359
# 150        | 124            | 0.89416
# 200        | 190            | 0.92769
# 250        | 244            | 0.94062

# 결론: 테스트한 epoch 중 최고 검증 성능을 기록한 epoch=250의 best.pt 를 predict 및 track 에 사용
# epoch=250 의 results.png 를 확인한 결과, 과적합의 뚜렷한 징후 없다고 판단함
# train loss 지표 전반적 감소 → 학습 데이터의 오차가 줄어듦
# 검증 mAP50-95 상승 → 검증 데이터에서도 탐지 성능이 개선됨
# val loss 정체 또는 감소 → 검증 오차가 악화되는 흐름이 아님
# 그래프는 검증 데이터에서의 성능을 보여주므로, 새로운 이미지에서도 탐지 결과가 좋은지 확인하면 됨
