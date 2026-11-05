"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict
import hashlib

torch.set_num_threads(1)
torch.manual_seed(14)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

rows = []
# defaultdict(list): 없는 키에 접근하면 빈 리스트 []를 자동으로 만들어 주는 딕셔너리
seen = defaultdict(list)
for split in ["train", "valid", "test"]:
    dataset = ImageFolder(DATA / split)
    for label, name in enumerate(dataset.classes):
        # TODO A01: 해당 라벨의 개수를 세세요.
        count = dataset.targets.count(label)

        # split의 클래스별 데이터 비율 확인
        rows.append({"split": split, "class": name, "n": count, "ratio": count/len(dataset)})
    for path,_ in dataset.samples:
        # 내용이 동일한 이미지의 상대 경로들을 해시값별로 묶어 중복 검사에 사용
        seen[hashlib.sha256(Path(path).read_bytes()).hexdigest()].append(str(Path(path).relative_to(DATA)))

table = pd.DataFrame(rows)
table.to_csv(OUT/"class_counts.csv", index=False)
print(table.round(4).to_string(index=False))

# 파일 내용이 동일한 이미지가 있는 경로 그룹을 추출 (split 내부는 물론 split 간 중복도 검사)
duplicates = [paths for paths in seen.values() if len(paths)>1]
print("바이트 동일 중복 그룹:",len(duplicates))
(OUT/"duplicates.json").write_text(json.dumps(duplicates,ensure_ascii=False,indent=2),encoding="utf-8")
# test 픽셀은 중복 감사에만 쓰며 모델 성능을 계산하지 않습니다.


#   누수가능성 설명
#   바이트 단위의 동일 중복 이미지는 발견되지 않았으나, 
#   Train·Valid·Test 간 시각적으로 유사한 이미지나 동일 원본에서 변형된 이미지가 포함되어 있을 경우 
#   데이터 누수(Data Leakage)가 발생할 가능성이 있다.

#   circle/square/triangle = 120/40/20개 (66.67%/22.22%/11.11%)로,
#   학습 데이터가 circle 클래스에 치우쳐 있음
#   해결 방안
#   1) 먼저 클래스 가중치를 적용한 손실 함수를 사용하는 것이 좋습니다. 적은 클래스의 오분류에 더 큰 손실을 부여하는 방법
#      valid 데이터의 클래스별 recall과 macro F1이 개선되는지 비교하세요. 가중치 손실과 균형 샘플링은 각각 먼저 실험하는 것이 효과를 판단하기 쉽습니다.
#   2) square, triangle 증강 적용 + 오버샘플링 적용