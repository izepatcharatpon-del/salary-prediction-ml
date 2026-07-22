# High-Salary Classification (Group G22)

End-to-end **binary classification** pipeline that predicts whether a person is in the **high-salary** group (`label` = 0/1) from demographic / employment features.

Course project · Mae Fah Luang University · Nov 2025  
Author: [izepatcharatpon-del](https://github.com/izepatcharatpon-del)

---

## Results (test set)

| Model | Test accuracy | Weighted F1 | Saved model |
| --- | ---: | ---: | --- |
| **ANN** (`MLPClassifier`) | **81.0%** | 0.810 | `artificial_neural_network/model/model_ann.joblib` |
| **KNN** (`n_neighbors=21`, GridSearch) | **79.9%** | 0.800 | `k_nearest_neighbor_4techniques/model/model_Knn.joblib` |
| **Random Forest** | **79.3%** | 0.793 | `random_forests/model/model_rf.joblib` |

Metrics from each model's evaluate notebook (`accuracy_score` + `classification_report` on held-out test features).

Example (ANN / MLP on test set):

```
Confusion Matrix:
[[2577  710]
 [ 384 2078]]

accuracy                          0.810
weighted avg F1                   0.811
```

---

## Pipeline

```
CSV → train/test/live split
    → clean missing values
    → encode (ordinal / one-hot)
    → scale numeric features
    → train KNN / RF / ANN
    → evaluate (accuracy, precision, recall, F1, confusion matrix)
    → joblib.dump → predict outside the notebook
```

Per-model notebooks follow: **process → train → (feature importance) → evaluate → live predict**.

---

## Inference outside notebooks (important)

Training lives in Jupyter. For demos / interviews, the saved `.joblib` models can be loaded from a plain Python script:

```bash
git lfs install
git clone https://github.com/izepatcharatpon-del/salary-prediction-ml.git
cd salary-prediction-ml
py -3 -m pip install -r requirements.txt

# Predict first 5 processed test rows with Random Forest
py -3 predict.py --model rf

# Or ANN / KNN
py -3 predict.py --model ann --n 10
py -3 predict.py --model knn --n 10
```

`predict.py` does: `joblib.load` → align columns to `model.feature_names_in_` → `model.predict`.

> Note: features must already be preprocessed the same way as training (see each model's `step2_process_data.ipynb`). A production follow-up would wrap preprocess + model in a single `sklearn.Pipeline`.

---

## Dataset

- Source: `Dataset/high_salary.csv` (~20.9k rows, census/employment-style features)
- Splits: `high_salary.train.csv` / `.test.csv` / `.live.csv`
- Target: `label` (0 = not high-salary, 1 = high-salary)
- Example features: age-group, workclass, education, occupation, hoursperweek, capital gain/loss, native-country, …

---

## Repo layout

```
Dataset/                         raw + split CSVs
step1_split_data.ipynb           shared train/test/live split
artificial_neural_network/       ANN notebooks + model_ann.joblib
k_nearest_neighbor_4techniques/  KNN + GridSearch notebooks + model_Knn.joblib
random_forests/                  RF notebooks + model_rf.joblib
predict.py                       load .joblib and predict outside notebooks
requirements.txt
```

---

## Stack

Python · pandas · scikit-learn · joblib · Jupyter  
(Originally run via Docker `jupyter/datascience-notebook`)

```bash
docker run -p 8888:8888 -v "${PWD}:/home/jovyan/work/ProjectML" jupyter/datascience-notebook
```

## Model files (Git LFS)

Trained `.joblib` files are stored with **Git LFS** (Random Forest alone is >100MB). After clone, if models look tiny, run:

```bash
git lfs pull
```

---

## Honest scope

This is a **course / lab end-to-end ML pipeline**, not a production salary product. Strengths shown: compare multiple models, proper train/test evaluation, persist models with joblib, and call them from a script outside the notebook.
