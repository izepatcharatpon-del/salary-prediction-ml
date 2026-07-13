# High-Salary Classification (Group G22)

End-to-end binary classification pipeline predicting high-salary labels from demographic/employment features.

## Dataset

- Source: `Dataset/high_salary.csv` (~20.9k rows)
- Splits: `high_salary.train.csv` / `.test.csv` / `.live.csv`
- Target: `label` (0/1)

## Models compared

| Folder | Model | Notes |
| --- | --- | --- |
| `k_nearest_neighbor_4techniques/` | KNN + GridSearch | saved as `model_Knn.joblib` |
| `random_forests/` | RandomForestClassifier | ~79% test accuracy |
| `artificial_neural_network/` | MLPClassifier (ANN) | ~81% test accuracy |

Pipeline per model: process data → train → (feature importance) → evaluate → live predict.

## Stack

Python, pandas, scikit-learn, joblib, Jupyter (originally run via Docker `jupyter/datascience-notebook`)

## How to run

1. Open notebooks in order starting from `step1_split_data.ipynb`
2. Or use Docker:

```bash
docker run -p 8888:8888 -v "${PWD}:/home/jovyan/work/ProjectML" jupyter/datascience-notebook
```

## Note on model files

`random_forests/model/model_rf.joblib` is excluded from this repo because it exceeds GitHub’s 100MB file limit. Re-run `random_forests/step3_train_model.ipynb` to regenerate it.
