# High-Salary Classification (Group G22)

End-to-end binary classification pipeline predicting high-salary labels from demographic/employment features.

## Dataset

- Source: `Dataset/high_salary.csv` (~20.9k rows)
- Splits: `high_salary.train.csv` / `.test.csv` / `.live.csv`
- Target: `label` (0/1)

## Models compared (test set, n=4792)

| Model | Test accuracy | Weighted F1 | Saved model |
| --- | ---: | ---: | --- |
| ANN (`MLPClassifier`) | **81.0%** | 0.810 | `model_ann.joblib` |
| KNN (`n_neighbors=21`, GridSearch) | **79.9%** | 0.800 | `model_Knn.joblib` |
| Random Forest | **79.3%** | 0.793 | `model_rf.joblib` |

Metrics come from each model’s evaluate notebook (`accuracy_score` + `classification_report` on `features.test.csv`).

Pipeline per model: process data → train → (feature importance) → evaluate → live predict.

## Stack

Python, pandas, scikit-learn, joblib, Jupyter (originally run via Docker `jupyter/datascience-notebook`)

## How to run

1. Open notebooks in order starting from `step1_split_data.ipynb`
2. Or use Docker:

```bash
docker run -p 8888:8888 -v "${PWD}:/home/jovyan/work/ProjectML" jupyter/datascience-notebook
```

## Model files (Git LFS)

Trained `.joblib` models are stored with **Git LFS** (Random Forest alone is >100MB).

```bash
git lfs install
git clone https://github.com/izepatcharatpon-del/G22-high-salary-ml.git
```
