import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommender import (load_data, make_pmf, make_user_cf, make_item_cf, evaluate_model, neighbor_sweep)

RESULTS = "/Users/aad8ya/Desktop/cse572-ass3/task-2/results"
PLOTS   = "/Users/aad8ya/Desktop/cse572-ass3/task-2/plots"

def run_2c(data):
    print("Q2c: evaluating PMF, User-CF, Item-CF ...")
    models = {
        "PMF":      make_pmf(),
        "User-CF":  make_user_cf("cosine", k=40),
        "Item-CF":  make_item_cf("cosine", k=40),
    }
    rows = []
    for name, algo in models.items():
        print(f"  {name}", flush=True)
        m = evaluate_model(algo, data)
        rows.append({"model": name, "MAE": m["mae"], "RMSE": m["rmse"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "q2c_metrics.csv"), index=False)

    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width/2 for i in x], df["MAE"],  width, label="MAE")
    ax.bar([i + width/2 for i in x], df["RMSE"], width, label="RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE by Model (5-fold CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q2c_mae_rmse.png"), dpi=150)
    plt.close()

    print(df.to_string(index=False))
    return df

def run_2d(data):
    csv = os.path.join(RESULTS, "q2c_metrics.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        df = run_2c(data)

    best_rmse = df.loc[df["RMSE"].idxmin(), "model"]
    best_mae  = df.loc[df["MAE"].idxmin(),  "model"]

    summary = pd.DataFrame([
        {"metric": "RMSE", "best_model": best_rmse,
         "value": df.loc[df["RMSE"].idxmin(), "RMSE"]},
        {"metric": "MAE",  "best_model": best_mae,
         "value": df.loc[df["MAE"].idxmin(),  "MAE"]},
    ])
    summary.to_csv(os.path.join(RESULTS, "q2d_comparison.csv"), index=False)

    print("\nQ2d: best model comparison")
    print(summary.to_string(index=False))
    print(f"\n  Best by RMSE: {best_rmse}")
    print(f"  Best by MAE:  {best_mae}")
    return summary

def run_2e(data):
    print("\nQ2e: similarity metric impact ...")
    sims = ["cosine", "msd", "pearson"]
    rows = []
    for sim in sims:
        for label, algo in [("User-CF", make_user_cf(sim, k=40)),
                             ("Item-CF", make_item_cf(sim, k=40))]:
            print(f"  {label} / {sim}", flush=True)
            m = evaluate_model(algo, data)
            rows.append({"model": label, "sim": sim,
                         "MAE": m["mae"], "RMSE": m["rmse"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "q2e_similarity.csv"), index=False)

    combos = [f"{r['model']}\n{r['sim']}" for _, r in df.iterrows()]
    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width/2 for i in x], df["MAE"],  width, label="MAE")
    ax.bar([i + width/2 for i in x], df["RMSE"], width, label="RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(combos, fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("MAE / RMSE by Model and Similarity Metric")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q2e_similarity_impact.png"), dpi=150)
    plt.close()

    print(df.to_string(index=False))
    return df

def run_2f(data):
    print("\nQ2f: neighbor count sweep (coarse) ...")
    k_values = list(range(1, 6, 4)) + list(range(5, 101, 5))
    k_values = sorted(set([1, 5, 10, 15, 20, 25, 30, 35, 40,
                            45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))

    df = neighbor_sweep(data, k_values)
    df.to_csv(os.path.join(RESULTS, "q2f_neighbors.csv"), index=False)

    for col, label in [("user", "User-CF"), ("item", "Item-CF")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["k"], df[f"{col}_rmse"], marker="o", label="RMSE")
        ax.plot(df["k"], df[f"{col}_mae"],  marker="s", label="MAE")
        ax.set_xlabel("K (neighbors)")
        ax.set_ylabel("Error")
        ax.set_title(f"{label}: Error vs. Number of Neighbors")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"q2f_neighbors_{col}.png"), dpi=150)
        plt.close()

    best_k_user = int(df.loc[df["user_rmse"].idxmin(), "k"])
    best_k_item = int(df.loc[df["item_rmse"].idxmin(), "k"])
    print(f"  Coarse best K - User-CF: {best_k_user}, Item-CF: {best_k_item}")
    print(df.to_string(index=False))
    return df

def run_2g(data):
    print("\nQ2g: fine-grained K sweep ...")
    csv = os.path.join(RESULTS, "q2f_neighbors.csv")
    if os.path.exists(csv):
        coarse = pd.read_csv(csv)
    else:
        coarse = run_2f(data)

    best_k_user = int(coarse.loc[coarse["user_rmse"].idxmin(), "k"])
    best_k_item = int(coarse.loc[coarse["item_rmse"].idxmin(), "k"])

    user_range = list(range(max(1, best_k_user - 10), best_k_user + 11))
    item_range = list(range(max(1, best_k_item - 10), best_k_item + 11))
    all_k = sorted(set(user_range + item_range))

    df = neighbor_sweep(data, all_k)
    df.to_csv(os.path.join(RESULTS, "q2g_fine.csv"), index=False)

    for col, label, fine_range in [
        ("user", "User-CF", user_range),
        ("item", "Item-CF", item_range),
    ]:
        subset = df[df["k"].isin(fine_range)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(subset["k"], subset[f"{col}_rmse"], marker="o", label="RMSE")
        ax.plot(subset["k"], subset[f"{col}_mae"],  marker="s", label="MAE")
        ax.set_xlabel("K (neighbors)")
        ax.set_ylabel("Error")
        ax.set_title(f"{label}: Fine-grained Error vs. K")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"q2g_fine_{col}.png"), dpi=150)
        plt.close()

    fine_best_user = int(df.loc[df["user_rmse"].idxmin(), "k"])
    fine_best_item = int(df.loc[df["item_rmse"].idxmin(), "k"])
    print(f"  Fine best K - User-CF: {fine_best_user}, Item-CF: {fine_best_item}")
    print(df.to_string(index=False))
    return df

def main():
    data = load_data()

    run_2c(data)
    run_2d(data)
    run_2e(data)
    run_2f(data)
    run_2g(data)

if __name__ == "__main__":
    main()
