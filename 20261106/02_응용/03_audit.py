"""실습 03. 정상 라벨의 면적·비율·클래스 수와 그룹 분리를 확인합니다."""
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
manifest=pd.read_csv(BASE/"data/manifest.csv")
rows=[]
# read_csv(): 숫자는 숫자형으로 자동 변환
# read_text(): 숫자도 문자형으로 읽음 -> int 변환 필요
for image in manifest.itertuples():
    for line in (BASE/"data"/image.label).read_text(encoding="utf-8").splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        # TODO A03: 정규화 면적 w*h와 원본 pixel 기준 가로세로비를 기록하세요.
        row = {
            "image_id": image.image_id,
            "split": image.split,
            "class_id": int(cls),
            "normalized_area": w*h,
            "box_aspect_ratio": (w * image.width) / (h * image.height)
        }
        rows.append(row)
r=pd.DataFrame(rows);r.to_csv(out/"label_audit.csv",index=False)

# 학습/검증/테스트 데이터의 클래스별 데이터수 확인
print(r.groupby(["split","class_id"]).size().unstack(fill_value=0).to_string())

# 같은 group_id가 둘 이상의 split에 걸쳐 있는 그룹의 개수를 구함. 데이터 분할 간 누수 가능성을 확인할 때 사용
# group_id는 서로 관련 있는 데이터를 묶는 식별자
print("split 간 그룹 중복:", int((manifest.groupby("group_id").split.nunique()>1).sum()))

# 중복데이터 검증
print("정확히 같은 파일:", int(manifest.sha256.duplicated().sum()))
print("빈 라벨은 정상 negative 이미지이며 누락 라벨과 다릅니다.")


''' 점검사항
    1) 학습·검증·테스트가 90:30:30개라면, 전체 150개에서 60% : 20% : 20% 이므로 무난한 분할, 실습용으로는 충분히 합리적인 구성
    2) 다만 전체가 150개이면 이미지 몇 장의 결과에 따라 평가 수치가 크게 달라질 수 있음
    3) 아래 확인이 필요
        - 각 split에 클래스가 적절히 포함되는지: 각 클래스도 학습·검증·테스트에 60% : 20% : 20%로 배분되므로, 클래스 분포 관점에서는 균형 잡힌 구성
        - 같은 group_id가 여러 split에 섞이지 않는지: split 간 그룹 중복 없음
        - 동일한 이미지가 중복으로 들어가지 않는지: 중복 이미지 없음
'''