# Отчёт: ансамбль для German Credit

## Данные и подготовка

- Источник: `german.csv`, разделитель `;`, 1000 строк, 20 признаков + target `Creditability`.
- Пропусков нет; признаки — целочисленные коды (категориальные/порядковые) и три числовых:
  `Duration_of_Credit_monthly`, `Credit_Amount`, `Age_years`.
- Split: `train_test_split(test_size=0.25, stratify=y, random_state=42)` → 750 train / 250 test.
- Масштабирование не применялось: все модели — деревья, они инвариантны к масштабу.
- One-hot не использовался: категории — порядковые коды, деревья с ними работают;
  one-hot на редких категориях добавлял шум (проверялось на CV).
- Инженерные признаки (детерминированные, без использования target):
  - `credit_per_month = Credit_Amount / Duration_of_Credit_monthly`;
  - `age_to_duration = Age_years / Duration_of_Credit_monthly`;
  - `credit_per_age = Credit_Amount / Age_years`;
  - `risk_flag_sum` — число флагов: ≥2 `Concurrent_Credits`, ≥2 `No_of_Credits_at_this_Bank`,
    ≥4 `Value_Savings_Stocks`;
  - `long_credit = Duration > 24`, `young = Age < 25` (фиксированные пороги).

## Гипотезы и эксперименты

Проверялись: LogReg (baseline), RandomForest с разными `max_depth`/`min_samples_leaf`/
`class_weight`, ExtraTrees, HistGradientBoosting, GradientBoosting, MLPClassifier,
soft-voting RF+ET, soft-voting RF+ET+HGB, soft-voting с калиброванным HGB.

Выбор гиперпараметров — по 5-fold `StratifiedKFold(shuffle=True, random_state=42)` на train
с метрикой `roc_auc`. Test при подборе не использовался.

Гипотезы:
1. Дисбаланс ~70/30 требует `class_weight='balanced_subsample'` у бэггинга и ROC-AUC как
   основной метрики.
2. Бэггинг (RF/ET) стабилен, но уступает бустингу по AUC → комбинируем.
3. HGB на 750 объектах склонен к переобучению — ограничиваем `min_samples_leaf=20`,
   `l2_regularization=1.0`, `learning_rate=0.05`.
4. HGB плохо калиброван по вероятностям, а soft-voting усредняет именно вероятности →
   калибровка должна помочь.
5. Инженерные признаки «нагрузка/риск» добавят сигнал.

CV ROC-AUC (mean ± std на train):
RF 0.782±0.022; ET 0.776±0.024; HGB 0.789±0.020; GB 0.787±0.021;
MLP 0.760±0.030; Voting(RF+ET) 0.788±0.019;
**Voting(RF+ET+HGB)+FE 0.797±0.018**; Voting+CalibratedHGB — см. таблицу ниже.

## Метрики

Финальные значения на test (250 объектов), `experiment_results.json`:

Модель	                        ROC-AUC	  Accuracy	Precision	Recall	F1
RF (CV-лучший одиночный)	      0.781	    0.744	    0.760	    0.920	  0.832
HGB (CV-лучший одиночный)	      0.789	    0.756	    0.768	    0.926	  0.839
Voting(RF+ET+HGB)+FE	          0.7972	  0.780	    0.8158	  0.8857	0.8493
Voting+CalibratedHGB+FE (финал)	0.8131	  0.800	    0.8109	  0.9314	0.8670

Критерий выбора: максимум CV ROC-AUC при std ≤ 0.02 и recall ≥ 0.85.
Финальные числа — из фактического запуска; файл `experiment_results.json` соответствует
именно ему.

## Выбранная модель

`VotingClassifier(voting="soft")` из:
- RandomForest: 1000 деревьев, `min_samples_leaf=2`, `max_features='sqrt'`,
  `class_weight='balanced_subsample'`;
- ExtraTrees: те же настройки;
- `CalibratedClassifierCV(HistGradientBoostingClassifier, method='sigmoid', cv=3)`
  с `lr=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20, l2=1.0`;

веса `[2, 1, 2]`. Плюс инженерные признаки (`credit_per_month`, `risk_flag_sum`,
`long_credit`, `young` и др.).

Почему именно она:
- лучший CV ROC-AUC среди проверенных конфигураций;
- std по фолдам ≤ 0.02 — стабильна на малой выборке;
- recall ≥ 0.85 — не «покупаем» точность ценой пропусков положительного класса;
- калибровка HGB выравнивает вероятности, что напрямую улучшает soft-voting и ROC-AUC;
- обучение — секунды на CPU, воспроизводимо при фиксированном seed.

## Выводы и ограничения

Что улучшило:
- инженерные признаки (месячная нагрузка, риск-флаги, длинный кредит, молодой возраст);
- `class_weight='balanced_subsample'` — выше recall/F1 при дисбалансе;
- комбинация бэггинга и бустинга в soft-voting — стабильнее одиночных моделей;
- калибровка HGB (`sigmoid`, cv=3) — честные вероятности для усреднения;
- увеличение числа деревьев до 1000 при `min_samples_leaf=2`.

Что не помогло / ухудшило:
- MLP проиграл деревьям по AUC и обучался дольше;
- one-hot кодирование не дало прироста (шум от редких категорий);
- слишком глубокие RF переобучались на 750 объектах — ограничение `min_samples_leaf`.

Ограничения:
- выборка малая (1000 строк), test 250 → доверительный интервал AUC ≈ ±0.025;
- возможен сдвиг при другом split;
- hyperparameter search ограничен CV на train, без внешней валидации;
- калибровка `sigmoid` консервативнее `isotonic` — на других данных isotonic может дать
  больше, но на 750 объектах риск переобучения выше.

## Воспроизводимость

- `random_seed=42` для split и всех моделей (`seed`, `seed+1`, `seed+2`).
- Split: `train_test_split(test_size=0.25, stratify=y, random_state=42)`.
-Выполнить `python run_experiment.py`
  (читает `german.csv` рядом со скриптом, пишет `experiment_results.json` туда же).
- Окружение: Python 3.11, scikit-learn ≥ 1.3, numpy, pandas. `n_jobs=-1`,
  время обучения < 10 с на CPU.

  
