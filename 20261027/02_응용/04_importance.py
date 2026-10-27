"""실습 04 응용: 전처리 출력 이름 8개와 중요도를 대응시킵니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# RandomForest는 거리나 벡터 크기 자체를 기준으로 판단하지 않고, “어디서 나눌지”를 기준으로 학습하기 때문에 보통 수치형 데이터를 정규화 할 필요없음
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.ensemble import RandomForestClassifier

model = Pipeline([("pre", clone(pre)), ("model", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))])
model.fit(X_train, y_train)
# TODO A04: 전처리의 출력 이름과 Forest의 중요도를 가져오세요.
# [오해주의] 문자형 열은 OneHotEncoder 를 거치므로 숫자형 열로 변환
# 전처리 후 특성 이름과 같은 순서의 중요도를 가져옴
names = model["pre"].get_feature_names_out()
values = model["model"].feature_importances_
# TODO 끝
print("names: ", names)
print("values: ", values)
result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
result.to_csv(OUT / "04_importance.csv", index=False, encoding="utf-8-sig")
print("변환된 특성 수:", len(names))

'''이번 실습 해석'''
# num__input_chars가 feature_importance_=0.5643 으로 분할 시 지니 불순도를 줄이는 데 가장 크게 기여했다
# 수치형 특성은 가능한 분할 기준이 많아 이런 중요도가 높게 나오는 편향이 있을 수 있습니다.
# 후보가 많으면 우연히 학습 데이터를 잘 나누는 기준을 찾을 기회도 많습니다. 그래서 불순도 감소로 계산하는 feature_importances_에서 중요도가 높게 나올 수 있다는 뜻입니다.
print(result.round(4).to_string(index=False))
print("중요도는 원인이나 신규 데이터 성능을 보장하지 않습니다.")
