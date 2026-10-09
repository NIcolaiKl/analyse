"""Графики для отчёта по лабораторной №2."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


# ---------- Titanic ----------

def titanic_plots() -> None:
    df = pd.read_csv(HERE / "titanic.csv")

    # 1. Пропуски по столбцам
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Пропуски по столбцам Titanic")
    ax.set_ylabel("Количество NaN")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT / "titanic_missing.png", dpi=150)
    plt.close(fig)

    # 2. Доля выживших по классу и полу
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex",
                errorbar=None, ax=ax)
    ax.set_title("Доля выживших по классу и полу")
    ax.set_ylabel("Survived (доля)")
    ax.set_xlabel("Pclass")
    plt.tight_layout()
    fig.savefig(OUT / "titanic_survival.png", dpi=150)
    plt.close(fig)

    # 3. Распределение возраста по классу
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Pclass", y="Age", ax=ax)
    ax.set_title("Возраст по классу")
    plt.tight_layout()
    fig.savefig(OUT / "titanic_age.png", dpi=150)
    plt.close(fig)

    # 4. Распределение Fare (лог-шкала — из-за тяжёлого хвоста)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Fare"].dropna(), bins=40, ax=ax, kde=True)
    ax.set_xscale("log")
    ax.set_title("Распределение Fare (log-шкала)")
    plt.tight_layout()
    fig.savefig(OUT / "titanic_fare.png", dpi=150)
    plt.close(fig)

    print(f"Titanic: графики сохранены в {OUT}")


# ---------- Brainsize ----------

def brainsize_plots() -> None:
    df = pd.read_csv(HERE / "brainsize.txt", sep="\t", na_values=["NA"])
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Scatter: MRI_Count vs каждый признак, цвет — пол
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, feat in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feat, y=TARGET, hue="Gender",
                        s=60, ax=ax, edgecolor="white")
        ax.set_title(f"{feat} vs {TARGET}")
        ax.set_xlabel(feat)
        ax.set_ylabel(TARGET if feat == FEATURES[0] else "")
    plt.tight_layout()
    fig.savefig(OUT / "brainsize_scatter.png", dpi=150)
    plt.close(fig)

    # 2. Heatmap корреляций отдельно для каждого пола
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, gender, title in zip(
        axes,
        ["Male", "Female"],
        ["Мужчины", "Женщины"],
    ):
        sub = df[df["Gender"] == gender][FEATURES + [TARGET]]
        sns.heatmap(
            sub.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"{title} (n={(df['Gender'] == gender).sum()})")
    plt.tight_layout()
    fig.savefig(OUT / "brainsize_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Сравнение корреляций с MRI_Count между полами
    men_r = (
        df[df["Gender"] == "Male"][FEATURES + [TARGET]]
        .corr()[TARGET].drop(TARGET)
    )
    women_r = (
        df[df["Gender"] == "Female"][FEATURES + [TARGET]]
        .corr()[TARGET].drop(TARGET)
    )
    comparison = pd.DataFrame({"Мужчины": men_r, "Женщины": women_r})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.plot(kind="barh", ax=ax, color=["steelblue", "indianred"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Корреляции признаков с {TARGET}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(OUT / "brainsize_corr_bar.png", dpi=150)
    plt.close(fig)

    # 4. Только топ-признак подробнее: scatter с линиями регрессии
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gender, title, color in zip(
        axes,
        ["Male", "Female"],
        ["Мужчины", "Женщины"],
        ["steelblue", "indianred"],
    ):
        sub = df[df["Gender"] == gender].dropna(subset=["VIQ", TARGET])
        sns.regplot(data=sub, x="VIQ", y=TARGET, ax=ax,
                    color=color, scatter_kws={"s": 60, "edgecolor": "white"})
        r = sub["VIQ"].corr(sub[TARGET])
        ax.set_title(f"{title}: VIQ vs {TARGET}  (r = {r:+.2f})")
        ax.set_xlabel("VIQ")
        ax.set_ylabel(TARGET)
    plt.tight_layout()
    fig.savefig(OUT / "brainsize_viq_regression.png", dpi=150)
    plt.close(fig)

    print(f"Brainsize: графики сохранены в {OUT}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="notebook")
    titanic_plots()
    brainsize_plots()
    print("Готово. Откройте папку:", OUT)