import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import (
    ClassificationInput,
    RegressionInput,
)
from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
    _metrics,
    _to_1d,
)

RNG = 42
HERE = Path(__file__).resolve().parent


# ------------------------- 1. Student Performance -------------------------

sp_path = HERE / "Student_Performance.csv"
if not sp_path.exists():
    raise FileNotFoundError(
        f"Не найден {sp_path}. Положите CSV рядом со скриптом "
        f"или поправьте путь."
    )

sp = pd.read_csv(sp_path)
sp["Extracurricular Activities"] = (
    sp["Extracurricular Activities"] == "Yes"
).astype(int)

X_reg = sp.drop(columns=["Performance Index"]).to_numpy(dtype=np.float64)
y_reg = sp["Performance Index"].to_numpy(dtype=np.float64)

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RNG
)

reg_in = RegressionInput(
    train_features=Xr_tr,
    train_target=yr_tr,
    test_features=Xr_te,
    test_target=yr_te,
    learning_rate=0.1,
    epochs=2000,
)
reg_res = train_linear_regression(reg_in)

y_test_pred = np.asarray(reg_res.predictions)
test_mse = float(np.mean((y_test_pred - yr_te) ** 2))
print(f"[LinearRegression] test MSE = {test_mse:.4f}")


# ------------------------- 2. German Credit -------------------------

gc_path = HERE / "german.csv"
if not gc_path.exists():
    raise FileNotFoundError(
        f"Не найден {gc_path}. Положите CSV рядом со скриптом "
        f"или поправьте путь."
    )

# Если в файле разделитель ';' — оставьте sep=";", иначе уберите его.
gc = pd.read_csv(gc_path, sep=";")

y_cls = gc.iloc[:, 0].to_numpy(dtype=np.int64)
X_cls = gc.iloc[:, 1:].to_numpy(dtype=np.float64)

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_cls, y_cls,
    test_size=0.2,
    random_state=RNG,
    stratify=y_cls,
)

cls_in = ClassificationInput(
    train_features=Xc_tr,
    train_target=yc_tr,
    test_features=Xc_te,
    test_target=yc_te,
    learning_rate=0.5,
    epochs=3000,
)
cls_res = train_logistic_classifier(cls_in)

own_proba = np.asarray(cls_res.probabilities)
own_pred = np.asarray(cls_res.predictions)
own_metrics = _metrics(_to_1d(yc_te), own_pred, own_proba)
print(f"[Own Logistic] {asdict(own_metrics)}")


# ------------------------- 3. Сравнение моделей -------------------------

cmp_res = compare_classical_models(cls_in)
print(f"[sklearn Logistic] {asdict(cmp_res.logistic_regression)}")
print(f"[DecisionTree]    {asdict(cmp_res.decision_tree)}")
print(f"[KNN]             {asdict(cmp_res.knn)}")


# ------------------------- Сохранение результатов -------------------------

results = {
    "linear_regression": {
        "test_mse": test_mse,
    },
    "classification": {
        "own_logistic": asdict(own_metrics),
        "logistic_regression": asdict(cmp_res.logistic_regression),
        "decision_tree": asdict(cmp_res.decision_tree),
        "knn": asdict(cmp_res.knn),
    },
}

out_path = HERE / "experiment_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nРезультаты сохранены в {out_path}")