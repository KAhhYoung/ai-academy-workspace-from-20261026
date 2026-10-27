"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
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
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))]
rows = []
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A03: model을 train으로 학습한 뒤 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
result = pd.DataFrame(rows)
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")




'''이번 실습 해석'''
# valid_F1을 기준으로 하면 Tree가 더 낫습니다.
# FP: 실제 성공인데 실패라고 예측 → Forest가 적음
# FN: 실제 실패인데 성공이라고 예측 → Tree가 적음

# 실패를 양성(1)으로 정의했을 때:
# - 불필요한 실패 경고를 줄이려면 → FP가 적은 모델(Forest)
# - 실패를 놓치지 않으려면 → FN이 적은 모델(Tree)

# 불필요한 실패 경고를 줄이는 것이 중요해서 FP가 적은 모델을 선택하려면 Forest. 그러나 valid_F1 이 더 높은 모델은 Tree
# 이럴 땐 어떤 기준으로 선택?
#    - 실제 목적에 맞는 기준을 우선합니다. F1이 항상 최우선 기준은 아닙니다.
#    - 이번 선택에는 다음 교환 관계가 있습니다.
#       Tree → Forest
#       불필요한 경고(FP): 4 → 2  (2건 감소)
#       놓친 실패(FN):     4 → 5  (1건 증가)
#    - 따라서 **“실패를 1건 더 놓치더라도 불필요한 경고를 2건 줄이는 것이 이득인가?”**로 판단하면 됩니다.
#    - 오류의 비용을 정할 수 있다면 다음처럼 비교합니다.
#       총비용 = FP * 불필요한_경고_1건의_비용 + FN * 실패를_놓친_1건의_비용
#      | FP 1건 비용 | FN 1건 비용 | Tree 총비용 | Forest 총비용 | 선택 |
#      |---:|---:|---:|---:|--------|
#      | 1  | 1  | 8  | 7  | Forest |
#      | 1  | 5  | 24 | 27 | Tree   |
#    - 비용을 정하기 어렵다면 **“놓친 실패는 최대 4건까지 허용하고, 그 안에서 FP가 가장 적은 모델을 선택”**처럼 제한을 정할 수도 있습니다.
#    - 최종 선택은 업무에서 어떤 오류를 얼마나 감수할 수 있는지에 따라 결정합니다.
print(result.round(3).to_string(index=False))
