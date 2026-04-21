import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import cross_validate

DATA_PATH = "/Users/aad8ya/Desktop/cse572-ass3/task-2/dataset/ratings_small.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

def sim_options(name, user_based):
    return {"name": name, "user_based": user_based}

def make_pmf():
    return SVD()

def make_user_cf(sim="cosine", k=40):
    return KNNWithMeans(k=k, sim_options=sim_options(sim, user_based=True))

def make_item_cf(sim="cosine", k=40):
    return KNNWithMeans(k=k, sim_options=sim_options(sim, user_based=False))

def evaluate_model(algo, data, n_folds=5):
    results = cross_validate(algo, data, measures=["MAE", "RMSE"], cv=n_folds, verbose=False)
    return {
        "mae": results["test_mae"].mean(),
        "rmse": results["test_rmse"].mean(),
    }

def neighbor_sweep(data, k_values, sim="cosine", n_folds=5):
    rows = []
    for k in k_values:
        print(f"  K={k}", flush=True)
        u = evaluate_model(make_user_cf(sim, k), data, n_folds)
        i = evaluate_model(make_item_cf(sim, k), data, n_folds)
        rows.append({"k": k,
                     "user_mae": u["mae"], "user_rmse": u["rmse"],
                     "item_mae": i["mae"], "item_rmse": i["rmse"]})
    return pd.DataFrame(rows)
