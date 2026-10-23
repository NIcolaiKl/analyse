"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def _to_frame(features) -> pd.DataFrame:
    """Приводим вход к DataFrame с числовыми колонками."""
    if isinstance(features, pd.DataFrame):
        return features.copy()
    return pd.DataFrame(np.asarray(features))


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Простые инженерные признаки, не использующие target.

    Все преобразования детерминированы и не зависят от других строк,
    поэтому их можно безопасно применять к train и test по отдельности.
    """
    out = df.copy()
    cols = out.columns.tolist()

    # Отношение суммы кредита к длительности (месячная нагрузка)
    if "Credit_Amount" in cols and "Duration_of_Credit_monthly" in cols:
        dur = out["Duration_of_Credit_monthly"].replace(0, np.nan)
        out["credit_per_month"] = (out["Credit_Amount"] / dur).fillna(0.0)

    # Возраст к длительности кредита
    if "Age_years" in cols and "Duration_of_Credit_monthly" in cols:
        dur = out["Duration_of_Credit_monthly"].replace(0, np.nan)
        out["age_to_duration"] = (out["Age_years"] / dur).fillna(0.0)

    # Сумма кредита на год возраста
    if "Credit_Amount" in cols and "Age_years" in cols:
        age = out["Age_years"].replace(0, np.nan)
        out["credit_per_age"] = (out["Credit_Amount"] / age).fillna(0.0)

    # Число «рисковых» флагов
    risk_flags = []
    if "Concurrent_Credits" in cols:
        risk_flags.append((out["Concurrent_Credits"] >= 2).astype(int))
    if "No_of_Credits_at_this_Bank" in cols:
        risk_flags.append((out["No_of_Credits_at_this_Bank"] >= 2).astype(int))
    if "Value_Savings_Stocks" in cols:
        risk_flags.append((out["Value_Savings_Stocks"] >= 4).astype(int))
    if risk_flags:
        out["risk_flag_sum"] = sum(risk_flags)

    # Бинарные индикаторы с фиксированными порогами (не зависят от данных)
    if "Duration_of_Credit_monthly" in cols:
        out["long_credit"] = (out["Duration_of_Credit_monthly"] > 24).astype(int)
    if "Age_years" in cols:
        out["young"] = (out["Age_years"] < 25).astype(int)

    return out


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    seed = int(getattr(data, "random_seed", 42))

    X_train = _add_features(_to_frame(data.train_features))
    X_test = _add_features(_to_frame(data.test_features))
    # Выравниваем колонки, чтобы не упасть на рассинхроне признаков
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = np.asarray(data.train_target).ravel()

    # --- Базовые модели ансамбля ---
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed + 1,
        n_jobs=-1,
    )

    # HGB плохо калиброван по вероятностям — оборачиваем в калибровку,
    # чтобы soft-voting усреднял честные вероятности.
    hgb_base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed + 2,
    )
    hgb = CalibratedClassifierCV(
        estimator=hgb_base,
        method="sigmoid",
        cv=3,
    )

    model = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        voting="soft",
        weights=[2, 1, 2],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    y_test = np.asarray(data.test_target).ravel()

    return EnsembleResult(
        model_name="Voting(RF+ET+CalibratedHGB, soft)",
        probabilities=proba,
        predictions=preds,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, preds)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
    )