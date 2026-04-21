import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmeans import load_data, euclidean, cosine_dist, jaccard_dist, kmeans, compute_accuracy

RESULTS = "/Users/aad8ya/Desktop/cse572-ass3/task-1/results"
PLOTS   = "/Users/aad8ya/Desktop/cse572-ass3/task-1/plots"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

METRICS = {
    "euclidean": euclidean,
    "cosine":    cosine_dist,
    "jaccard":   jaccard_dist,
}
K = 10
SEEDS = list(range(5))

def run_trials(data, labels, metric_name, stop_criteria, max_iter=500):
    fn = METRICS[metric_name]
    results = []
    for s in SEEDS:
        r = kmeans(data, K, fn, stop_criteria, max_iter=max_iter, seed=s)
        acc = compute_accuracy(r["assignments"], labels, K)
        results.append({
            "seed":        s,
            "sse":         r["sse_history"][-1],
            "accuracy":    acc,
            "iterations":  r["iterations"],
            "time":        r["time_elapsed"],
            "stop_reason": r["stop_reason"],
        })
    return results

def avg(trials, key):
    return np.mean([t[key] for t in trials])

def bar_chart(names, values, ylabel, title, path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def q1_q2_q3(data, labels):
    print("\n=== Q1 / Q2 / Q3 ===")
    rows = []
    for name in METRICS:
        print(f"  running {name} (combined, max_iter=500) ...")
        trials = run_trials(data, labels, name, "combined", max_iter=500)
        rows.append({
            "metric":     name,
            "avg_sse":    avg(trials, "sse"),
            "avg_acc":    avg(trials, "accuracy"),
            "avg_iter":   avg(trials, "iterations"),
            "avg_time_s": avg(trials, "time"),
        })

    df = pd.DataFrame(rows).set_index("metric")

    df[["avg_sse"]].to_csv(f"{RESULTS}/q1_sse.csv")
    df[["avg_acc"]].to_csv(f"{RESULTS}/q2_accuracy.csv")
    df[["avg_iter", "avg_time_s"]].to_csv(f"{RESULTS}/q3_convergence.csv")

    bar_chart(df.index, df["avg_sse"], "SSE", "Average SSE by Metric",
              f"{PLOTS}/q1_sse_comparison.png")
    bar_chart(df.index, df["avg_acc"], "Accuracy", "Average Accuracy by Metric",
              f"{PLOTS}/q2_accuracy_comparison.png", color="darkorange")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df.index, df["avg_iter"], color="steelblue")
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Convergence: Iterations")
    axes[1].bar(df.index, df["avg_time_s"], color="seagreen")
    axes[1].set_ylabel("Wall-clock time (s)")
    axes[1].set_title("Convergence: Time")
    plt.tight_layout()
    fig.savefig(f"{PLOTS}/q3_convergence.png", dpi=150)
    plt.close(fig)

    print("\nQ1 — Average SSE:")
    print(df[["avg_sse"]].to_string())
    print("\nQ2 — Average Accuracy:")
    print(df[["avg_acc"]].to_string())
    print("\nQ3 — Convergence:")
    print(df[["avg_iter", "avg_time_s"]].to_string())

    return df

def q4(data, labels):
    print("\n=== Q4 ===")
    criteria = {
        "centroid": dict(stop_criteria="centroid",  max_iter=500),
        "sse":      dict(stop_criteria="sse",       max_iter=500),
        "max_iter": dict(stop_criteria="max_iter",  max_iter=100),
    }

    rows = []
    for name in METRICS:
        for crit, kwargs in criteria.items():
            print(f"  running {name} / {crit} ...")
            trials = run_trials(data, labels, name, **kwargs)
            rows.append({
                "metric":    name,
                "criterion": crit,
                "avg_sse":   avg(trials, "sse"),
                "avg_iter":  avg(trials, "iterations"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/q4_sse_criteria.csv", index=False)

    pivot = df.pivot(index="metric", columns="criterion", values="avg_sse")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot))
    w = 0.25
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * w, pivot[col], w, label=col)
    ax.set_xticks(x + w)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("SSE")
    ax.set_title("SSE by Metric and Stop Criterion")
    ax.legend(title="Stop criterion")
    plt.tight_layout()
    fig.savefig(f"{PLOTS}/q4_sse_by_criteria.png", dpi=150)
    plt.close(fig)

    print("\nQ4 — SSE by metric x criterion:")
    print(pivot.to_string())

    return df

if __name__ == "__main__":
    print("Loading data...")
    data, labels = load_data()
    print(f"  data shape: {data.shape}, labels shape: {labels.shape}")

    q1_q2_q3(data, labels)
    q4(data, labels)

    print("\nDone. Results in task-1/results/, plots in task-1/plots/")
