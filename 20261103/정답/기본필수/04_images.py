"""실습 04 · 필요한 하위 폴더는 ImageFolder의 이미지 저장 구조뿐이다."""
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parent
# Resize(size)는 목표 크기가 필수. ToTensor()에는 필수 인자가 없다.
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
# ImageFolder(root, transform=None): root는 필수, transform은 선택이다.
# BEGIN B04


# ImageFolder: 폴더 구조를 기반으로 이미지 데이터와 클래스 레이블을 자동으로 불러오는 데이터셋 클래스
# root              : 이미지 데이터가 저장된 최상위 폴더 경로 (필수 인자)
#                     예: data/train/horizontal/, data/train/vertical/
# transform         : 이미지에 적용할 전처리 함수 (선택 인자, 기본값 None)
#                     여기서는 Resize(32, 32)와 ToTensor()를 순서대로 적용
# target_transform  : 정답 레이블(Label)에 적용할 변환 함수 (선택 인자, 기본값 None)
# loader            : 이미지 파일을 읽어오는 함수 (기본값 default_loader, 일반적으로 RGB 이미지로 로드)
# is_valid_file     : 파일이 유효한 이미지인지 판별하는 사용자 정의 함수 (기본값 None)
# allow_empty       : 이미지가 없는 클래스 폴더를 허용할지 설정 (기본값 False)
#
# ImageFolder는 하위 폴더 이름을 기준으로 클래스를 자동 분류하고 정수 레이블을 부여합니다.
# 예: horizontal → 0, vertical → 1 (폴더 이름의 알파벳 정렬 순서 기준)
# 이미지 한 개의 출력 형태: (이미지 텐서, 정수 레이블)
# RGB 이미지 기준 텐서 Shape: (3, 32, 32)
train = datasets.ImageFolder(root / "data" / "train", transform=transform)
# END B04
loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
images, labels = next(iter(loader))
print("학습 이미지:", len(train))
print("클래스:", train.class_to_idx)
print("배치:", tuple(images.shape), tuple(labels.shape))
print("라벨:", labels.tolist())
