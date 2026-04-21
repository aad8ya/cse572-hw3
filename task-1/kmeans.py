import time
import numpy as np
import pandas as pd


def load_data():
    base = "/Users/aad8ya/Desktop/cse572-ass3/task-1/dataset"
    X = pd.read_csv(f"{base}/data.csv", header=None).values.astype(np.float64)
    y = pd.read_csv(f"{base}/label.csv", header=None).values.ravel()
    return X, y

def euclidean(a, b):
    diff = a - b
    return np.sqrt(diff @ diff)

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (a @ b) / (na * nb)

def jaccard_dist(a, b):
    denom = np.sum(np.maximum(a, b))
    if denom == 0:
        return 0.0
    return 1.0 - np.sum(np.minimum(a, b)) / denom

def _euclidean_matrix(X, centroids):
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    C2 = np.sum(centroids ** 2, axis=1, keepdims=True).T
    cross = X @ centroids.T
    d2 = X2 + C2 - 2 * cross
    return np.sqrt(np.maximum(d2, 0.0))

def _cosine_matrix(X, centroids):
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    Cn = np.linalg.norm(centroids, axis=1, keepdims=True).T
    dot = X @ centroids.T
    norms = Xn * Cn
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(norms == 0, 0.0, dot / norms)
    return 1.0 - sim

def _jaccard_matrix(X, centroids):
    n, k = len(X), len(centroids)
    mins = np.empty((n, k))
    maxs = np.empty((n, k))
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        Xc = X[start:end, np.newaxis, :]
        Ce = centroids[np.newaxis, :, :]
        mins[start:end] = np.minimum(Xc, Ce).sum(axis=2)
        maxs[start:end] = np.maximum(Xc, Ce).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(maxs == 0, 0.0, 1.0 - mins / maxs)
    return jac

_MATRIX_FN = {
    "euclidean":  _euclidean_matrix,
    "cosine_dist": _cosine_matrix,
    "jaccard_dist": _jaccard_matrix,
}

def _get_matrix_fn(distance_fn):
    return _MATRIX_FN.get(distance_fn.__name__, None)

def _assign(X, centroids, distance_fn):
    matrix_fn = _get_matrix_fn(distance_fn)
    if matrix_fn is not None:
        dists = matrix_fn(X, centroids)
    else:
        n, k = len(X), len(centroids)
        dists = np.empty((n, k))
        for j, c in enumerate(centroids):
            dists[:, j] = np.array([distance_fn(x, c) for x in X])
    return np.argmin(dists, axis=1), dists

def _update_centroids(X, assignments, k, rng):
    d = X.shape[1]
    new_centroids = np.empty((k, d))
    for j in range(k):
        mask = assignments == j
        if np.any(mask):
            new_centroids[j] = X[mask].mean(axis=0)
        else:
            new_centroids[j] = X[rng.integers(len(X))]
    return new_centroids

def _sse(dists, assignments):
    point_dists = dists[np.arange(len(assignments)), assignments]
    return float(np.sum(point_dists ** 2))

def kmeans(data, k, distance_fn, stop_criteria="combined", max_iter=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=k, replace=False)
    centroids = data[idx].copy()

    assignments = np.zeros(len(data), dtype=int)
    sse_history = []
    stop_reason = "max_iter"
    t0 = time.time()

    for it in range(max_iter):
        new_assignments, dists = _assign(data, centroids, distance_fn)
        new_centroids = _update_centroids(data, new_assignments, k, rng)
        sse = _sse(dists, new_assignments)
        sse_history.append(sse)

        centroid_unchanged = np.allclose(centroids, new_centroids)
        sse_increased = len(sse_history) > 1 and sse > sse_history[-2]

        assignments = new_assignments
        centroids = new_centroids

        if stop_criteria == "centroid" and centroid_unchanged:
            stop_reason = "centroid"
            break
        elif stop_criteria == "sse" and sse_increased:
            stop_reason = "sse"
            break
        elif stop_criteria == "max_iter" and (it + 1) >= max_iter:
            stop_reason = "max_iter"
            break
        elif stop_criteria == "combined" and (centroid_unchanged or sse_increased):
            stop_reason = "centroid" if centroid_unchanged else "sse"
            break

    return {
        "centroids":   centroids,
        "assignments": assignments,
        "sse_history": sse_history,
        "iterations":  len(sse_history),
        "time_elapsed": time.time() - t0,
        "stop_reason": stop_reason,
    }

def compute_accuracy(assignments, true_labels, k):
    correct = 0
    for j in range(k):
        mask = assignments == j
        if not np.any(mask):
            continue
        _, counts = np.unique(true_labels[mask], return_counts=True)
        correct += counts.max()
    return correct / len(true_labels)
