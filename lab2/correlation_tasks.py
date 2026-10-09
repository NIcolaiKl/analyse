"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import (
    BrainCorrelationSummary,
    BrainDataInput,
)


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"
NUMERIC_COLS = FEATURES + [TARGET]


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Анализ brainsize.txt: корреляции признаков с MRI_Count по полу."""
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])


    # Разделяем по полу
    men_df = df[df["Gender"] == "Male"]
    women_df = df[df["Gender"] == "Female"]

    # Количество наблюдений в каждой группе
    men_count = int(len(men_df))
    women_count = int(len(women_df))

    def corr_with_mri(sub: pd.DataFrame) -> dict[str, float]:
        """Pearson r каждого признака с MRI_Count внутри группы."""
        return {
            feat: float(sub[feat].corr(sub[TARGET], method="pearson"))
            for feat in FEATURES
        }

    men_mri_correlation = corr_with_mri(men_df)
    women_mri_correlation = corr_with_mri(women_df)

    # Признак с максимальным |r| среди обеих групп
    best_per_feature = {
        feat: max(
            abs(men_mri_correlation[feat]) if pd.notna(men_mri_correlation[feat]) else 0.0,
            abs(women_mri_correlation[feat]) if pd.notna(women_mri_correlation[feat]) else 0.0,
        )
        for feat in FEATURES
    }
    strongest_mri_feature = max(best_per_feature, key=best_per_feature.get)

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).parent

    txt_path = here / "brainsize.txt"

    summary = analyze_brain_correlations(BrainDataInput(csv_path=str(txt_path)))

    print("=== Brainsize ===")
    print(f"Файл: {txt_path.name}")
    print(f"Мужчин: {summary.men_count}, женщин: {summary.women_count}")
    print()
    print(f"{'Признак':10s} {'Male':>10s} {'Female':>10s} {'max|r|':>10s}")
    for feat in FEATURES:
        m = summary.men_mri_correlation[feat]
        f = summary.women_mri_correlation[feat]
        mx = max(abs(m), abs(f))
        print(f"{feat:10s} {m:>+10.3f} {f:>+10.3f} {mx:>10.3f}")
    print(f"\nstrongest_mri_feature: {summary.strongest_mri_feature}")