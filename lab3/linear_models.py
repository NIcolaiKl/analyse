from __future__ import annotations

from typing import List, Tuple

import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


# ----------------------- утилиты -----------------------

def _to_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(-1)


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _bce(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------- 1. Линейная регрессия -----------------------

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = _to_2d(data.train_features)
    y_train = _to_1d(data.train_target)
    X_test = _to_2d(data.test_features)

    X_train_s, X_test_s = _standardize(X_train, X_test)
    Xb_train = _add_bias(X_train_s)
    Xb_test = _add_bias(X_test_s)

    n, d = Xb_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)

    w = np.zeros(d, dtype=np.float64)
    history: List[float] = []

    for epoch in range(epochs):
        y_pred = Xb_train @ w
        err = y_pred - y_train
        grad = (2.0 / n) * (Xb_train.T @ err)
        w -= lr * grad

        # считаем MSE на каждом шаге (недорого и удобно для отчёта)
        history.append(_mse(y_train, y_pred))

    predictions = Xb_test @ w

    return RegressionResult(
        weights=w[1:].copy(),
        bias=float(w[0]),
        loss_history=history,
        predictions=predictions.tolist(),
    )


# ----------------------- 2. Логистическая регрессия -----------------------

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = _to_2d(data.train_features)
    y_train = _to_1d(data.train_target).astype(int)
    X_test = _to_2d(data.test_features)

    X_train_s, X_test_s = _standardize(X_train, X_test)
    Xb_train = _add_bias(X_train_s)
    Xb_test = _add_bias(X_test_s)

    n, d = Xb_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)
    l2 = 1e-4  # маленькая L2-регуляризация для устойчивости

    w = np.zeros(d, dtype=np.float64)
    history: List[float] = []

    for epoch in range(epochs):
        z = Xb_train @ w
        p = _sigmoid(z)
        grad = (1.0 / n) * (Xb_train.T @ (p - y_train))
        grad[1:] += l2 * w[1:]
        w -= lr * grad
        history.append(_bce(y_train, p))

    probs = _sigmoid(Xb_test @ w)
    preds = (probs >= 0.5).astype(int)

    return ClassificationResult(
        weights=w[1:].copy(),
        bias=float(w[0]),
        loss_history=history,
        probabilities=probs.tolist(),
        predictions=preds.tolist(),
    )


# ----------------------- 3. Сравнение моделей -----------------------

def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # усреднение рангов при совпадениях
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1

    n_pos, n_neg = len(pos), len(neg)
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> ClassificationMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClassificationMetrics(
        roc_auc=_roc_auc(y_true, scores),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнение LogisticRegression / DecisionTree / KNN из sklearn
    на том же train/test, что и собственная модель."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    X_train = _to_2d(data.train_features)
    y_train = _to_1d(data.train_target).astype(int)
    X_test = _to_2d(data.test_features)
    y_test = _to_1d(data.test_target).astype(int)

    X_train_s, X_test_s = _standardize(X_train, X_test)

    # LogisticRegression
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    proba_lr = lr.predict_proba(X_test_s)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)
    m_lr = _metrics(y_test, pred_lr, proba_lr)

    # DecisionTree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_s, y_train)
    pred_dt = dt.predict(X_test_s)
    proba_dt = dt.predict_proba(X_test_s)[:, 1]
    m_dt = _metrics(y_test, pred_dt, proba_dt)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    pred_knn = knn.predict(X_test_s)
    proba_knn = knn.predict_proba(X_test_s)[:, 1]
    m_knn = _metrics(y_test, pred_knn, proba_knn)

    return BaselineComparisonResult(
        logistic_regression=m_lr,
        decision_tree=m_dt,
        knn=m_knn,
    )