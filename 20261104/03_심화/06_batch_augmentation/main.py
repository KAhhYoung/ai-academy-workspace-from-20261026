from pathlib import Path
import os
import cv2
import numpy as np
import csv
import json
from time import perf_counter
import hashlib

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

Path("outputs").mkdir(exist_ok=True)


def augment(image: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    '''
    타입힌트 및 함수 정의 작성하면 호버 시 정리된 내용이 나타남 
    입력값:

    증강 방식:
    1.
    2.

    '''
    rng=np.random.default_rng(seed)
    
    flip=bool(rng.random()<0.5)
    beta=int(rng.integers(-20,21))

    # flip만 해도 ok, beta만 더해도 ok, 둘 다 적용해도 ok

    # flipCode: 0(상하 뒤집기)
    #           양수,보통1(좌우 뒤집기)
    #           음수,보통-1(상하·좌우 모두 뒤집기 → 180도 회전과 같은 결과) 
    result=cv2.flip(src=image,flipCode=1) if flip else image.copy()
    # >>> TODO beta를 적용하되 uint8의 순환을 막고 값 범위를 지키세요. 원본 image는 수정하지 않습니다.
    # np.int16 의 표현범위: 부호가 있는 16비트 정수형; -32,768 ~ 32,767
    # uint8 그대로 계산: 250 + 20 → 14로 순환 → clip(14) → 14 ❌
    # int16으로 바꾼 뒤 계산: 250 + 20 → 270 유지 → clip(270) → 255 ✅
    result = np.clip(result.astype(np.int16) + beta, a_min=0, a_max=255).astype(np.int8)

    return result,{"seed":seed,"flip":flip,"beta":beta}

# 데이터 메타데이터 불러오기
with open("../data/manifest.csv",encoding="utf-8-sig",newline="") as f:
    sources=list(csv.DictReader(f))

rows=[]
for row in sorted(sources,key=lambda r:r["sample_id"]):
    if row["split"]!="train":
        continue  # valid/test는 증강 대상이 아닙니다.

    '''학습 데이터에 대해서만 다음을 수행'''

    # seed 를 데이터마다 다르게 설정하는 이유:
    # 원본 데이터가 다르면, 같은 seed로 같은 변환을 적용해도 최종 이미지는 다릅니다.
    # 구분할 것은 변환 설정과 변환 결과 이미지입니다.
    # 데이터 1 + seed 42 → 회전 +7도 → 회전된 데이터 1
    # 데이터 2 + seed 42 → 회전 +7도 → 회전된 데이터 2
    # 각 이미지 처리 시작 시 같은 seed로 난수 생성기를 초기화하고 같은 순서로 난수를 사용한다면, 선택한 각도는 같지만 원본이 달라 결과 이미지는 다릅니다.
    # 그럼에도 데이터마다 seed를 다르게 주려는 목적은 원본뿐 아니라 변환 설정도 다양하게 만들기 위해서입니다.
    seed=42+int(row["sample_id"].split("_")[-1])

    # 학습이미지 불러오기
    src=Path("../data")/row["path"];
    image=cv2.imread(str(src))

    result,settings=augment(image,seed);

    # 증강한 이미지를 저장
    dest=Path("outputs")/(row["sample_id"]+"_aug.png")
    saved=cv2.imwrite(str(dest),result)

    rows.append([row["sample_id"],row["split"],str(dest),seed,settings["flip"],settings["beta"],saved,hashlib.sha256(src.read_bytes()).hexdigest()])

with open("outputs/augmentation.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);
    w.writerow(["id","split","output","seed","flip","beta","saved","source_sha256"]);
    w.writerows(rows)

print("증강 이미지:",len(rows),"/ train만 처리")
print("설정:",[(r[0],r[3],r[4],r[5]) for r in rows])
