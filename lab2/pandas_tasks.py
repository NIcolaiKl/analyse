"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Загрузка и анализ датасета Titanic."""
    df = pd.read_csv(data.csv_path)

    # 1. Количество строк
    row_count = int(len(df))

    # 2. Пропуски по каждому столбцу
    missing_by_column = df.isna().sum().to_dict()

    # 3. Пассажиры старше 30 лет (NaN не считаются)
    adults_over_30_count = int((df["Age"] > 30).sum())

    # 4. Средний возраст по классам (NaN игнорируются)
    mean_age_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Age"].mean().to_dict().items()
    }

    # 5. Доля выживших по классам
    survival_rate_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Survived"].mean().to_dict().items()
    }

    # 6. Пять наибольших тарифов в порядке убывания
    highest_fares = (
        df["Fare"].dropna().sort_values(ascending=False).head(5).tolist()
    )

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).parent

    # Ищем CSV: сначала titanic.csv, потом любой *.csv рядом со скриптом
    csv_path = here / "titanic.csv"
    summary = analyze_titanic(TitanicInput(csv_path=str(csv_path)))

    print("=== Titanic ===")
    print(f"Файл: {csv_path.name}")
    print(f"Строк всего: {summary.row_count}")
    print("Пропуски по столбцам:")
    for col, cnt in summary.missing_by_column.items():
        print(f"  {col:12s}: {cnt}")
    print(f"Пассажиров старше 30: {summary.adults_over_30_count}")
    print("Средний возраст по Pclass:")
    for k, v in summary.mean_age_by_pclass.items():
        print(f"  {k}: {v:.2f}")
    print("Доля выживших по Pclass:")
    for k, v in summary.survival_rate_by_pclass.items():
        print(f"  {k}: {v:.4f}")
    print(f"Топ-5 Fare: {summary.highest_fares}")