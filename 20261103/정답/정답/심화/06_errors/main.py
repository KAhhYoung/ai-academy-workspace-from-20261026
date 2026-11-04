
import torch
from torch import nn
from pathlib import Path
import sys
import pandas as pd

torch.set_num_threads(1)
torch.manual_seed(42)

# ============================================================
# 1. GPU(CUDA) 사용 환경 설정
# ============================================================

# NVIDIA CUDA GPU가 사용 가능하면 cuda:0 사용
# GPU가 없거나 CUDA를 사용할 수 없으면 CPU로 자동 전환
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

if device.type == "cuda":
    print("GPU 이름:", torch.cuda.get_device_name(0))


# ============================================================
# 2. 프로젝트 경로 설정 및 모듈 불러오기
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# 제공 모듈의 경로를 Python 모듈 검색 경로에 추가
sys.path.insert(0, str(ROOT / "day12_sequence_models"))

# 결과 저장 디렉터리 생성
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

from models import classifier       # type: ignore
from data import classification_data
from train import fit               # type: ignore


# ============================================================
# 3. Tensor를 GPU로 이동하는 함수
# ============================================================

def move_to_device(obj, device):
    """
    데이터 구조 안에 포함된 Tensor를 GPU 또는 CPU로 이동합니다.

    - Tensor: .to(device)로 이동
    - dict: 내부 값들을 재귀적으로 이동
    - list/tuple: 내부 Tensor를 재귀적으로 이동
    - 문자열, 숫자 등: 그대로 유지
    """

    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, dict):
        return {
            k: move_to_device(v, device)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


# ============================================================
# 4. 데이터 준비
# ============================================================

# 학습/검증 데이터를 생성
data = classification_data(ROOT / "data")

# 학습에 사용할 Tensor를 GPU로 이동
gpu_data = move_to_device(data, device)

# 모델별 비교 결과와 예측 결과를 저장할 리스트
tables = []
rows = []


# ============================================================
# 5. RNN / LSTM / Transformer 학습 및 비교
# ============================================================

for name in ["RNN", "LSTM", "Transformer"]:

    # 모델별 가중치 초기화 조건을 일정하게 유지
    torch.manual_seed(42)

    # --------------------------------------------------------
    # 모델 생성 및 GPU 이동
    # --------------------------------------------------------

    # classifier(name): 모델 이름에 맞는 신경망 생성
    # .to(device): 모델의 파라미터와 버퍼를 GPU로 이동
    model = classifier(name).to(device)

    print(f"\n[{name}] 학습 시작")
    print("모델 장치:", next(model.parameters()).device)

    # --------------------------------------------------------
    # 모델 학습
    # --------------------------------------------------------

    # fit() 함수가 학습 및 검증을 수행
    # GPU로 이동한 모델과 데이터를 전달
    #
    # epochs=25 : 전체 학습 데이터를 25번 반복 학습
    # lr=0.01   : Optimizer의 학습률
    #
    # 주의: fit() 내부에서 CPU 이동을 강제하지 않아야 함
    result = fit(
        model,
        gpu_data,
        "classification",
        epochs=25,
        lr=0.01
    )

    # --------------------------------------------------------
    # 예측 결과에서 클래스 라벨 추출
    # --------------------------------------------------------

    # prediction은 클래스별 Logits
    # 예: [0.2, 1.8] -> 클래스 1
    #     [2.1, 0.4] -> 클래스 0
    #
    # argmax(dim=1):
    # 각 데이터에서 가장 높은 Logit의 인덱스를 선택
    #
    # detach(): Gradient 계산 그래프에서 분리
    # cpu()   : GPU Tensor를 CPU로 이동
    # tolist(): Python 리스트로 변환
    pred = (
        result["prediction"]
        .argmax(dim=1)
        .detach()
        .cpu()
        .tolist()
    )

    # 실제 정답은 원본 데이터에서 가져옴
    actual = data["valid"]["y"]

    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().tolist()

    # --------------------------------------------------------
    # 실제 정답과 예측값 비교
    # --------------------------------------------------------

    # 각 검증 데이터마다 다음 정보 기록
    # model      : 사용한 모델
    # id         : 데이터 식별자
    # actual     : 실제 정답
    # prediction : 모델 예측값
    # error      : 예측이 틀렸는지 여부
    tables.extend(
        {
            "model": name,
            "id": i,
            "actual": int(y),
            "prediction": int(p),
            "error": bool(y != p)
        }
        for i, y, p in zip(
            data["valid"]["ids"],
            actual,
            pred
        )
    )

    # --------------------------------------------------------
    # 모델별 성능 비교 정보 저장
    # --------------------------------------------------------

    rows.append({
        "model": name,
        "params": result["params"],
        "seconds": result["seconds"],
        "valid_accuracy": result["valid_accuracy"]
    })


# ============================================================
# 6. 학습 및 검증 결과 저장
# ============================================================

# 데이터별 실제 정답/예측값/오류 여부 저장
pd.DataFrame(tables).to_csv(
    OUT / "predictions.csv",
    index=False
)

# RNN / LSTM / Transformer 성능 비교 결과 저장
pd.DataFrame(rows).to_csv(
    OUT / "comparison.csv",
    index=False
)


# ============================================================
# 7. 최종 결과 출력
# ============================================================

# 모델별 예측 오류 개수를 출력
print(
    pd.DataFrame(tables)
    .groupby("model")["error"]
    .sum()
    .to_string()
)

print("\n모델별 성능 비교")
print(pd.DataFrame(rows).to_string(index=False))

print(
    "\n오류가 0이면 0건으로 기록한다. "
    "오류 예시를 꾸며내지 않는다."
    "실제로 작은 모델에서는 오류가 발생하지 않을 수 있습니다. 그래도 완전히 오류가 없다고 판단하면 안돼요."
)
