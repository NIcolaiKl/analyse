"""Запуск эксперимента и сохранение метрик в experiment_results.json."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "german.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {CSV_PATH}. Положите german.csv рядом с run_experiment.py "
        f"или поправьте CSV_PATH."
    )

df = pd.read_csv(CSV_PATH, sep=";")
X = df.drop(columns=["Creditability"])
y = df["Creditability"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

res = run_ensemble_experiment(
    EnsembleInput(
        train_features=X_tr,
        train_target=y_tr,
        test_features=X_te,
        test_target=y_te,
        random_seed=42,
    )
)

out = {
    "model_name": res.model_name,
    "roc_auc": round(res.roc_auc, 4),
    "accuracy": round(res.accuracy, 4),
    "precision": round(res.precision, 4),
    "recall": round(res.recall, 4),
    "f1": round(res.f1, 4),
    "split": {
        "test_size": 0.25,
        "stratify": True,
        "random_seed": 42,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    },
    "features": (
        "raw + credit_per_month + age_to_duration + credit_per_age + "
        "risk_flag_sum + long_credit + young"
    ),
    "model": (
        "VotingClassifier(soft): "
        "RF(1000, leaf=2, balanced_subsample) + "
        "ET(1000, leaf=2, balanced_subsample) + "
        "CalibratedHGB(sigmoid, cv=3, lr=0.05, iter=400, leaves=31), "
        "weights=[2,1,2]"
    ),
}

out_path = BASE / "experiment_results.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print(f"\nСохранено в {out_path}")