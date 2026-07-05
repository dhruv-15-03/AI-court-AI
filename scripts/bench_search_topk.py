import numpy as np, time

def old_topk(scores, k):
    return np.argsort(-scores)[:k]

def new_topk(scores, k):
    n = scores.shape[0]
    if k >= n:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]

rng = np.random.default_rng(42)
for n in (1000, 10000, 100000, 500000):
    scores = rng.random(n).astype(np.float64)
    k = 10
    # correctness check
    a = old_topk(scores, k)
    b = new_topk(scores, k)
    assert np.array_equal(scores[a], scores[b]), f"mismatch at n={n}"

    reps = 50 if n <= 100000 else 10
    t0 = time.perf_counter()
    for _ in range(reps):
        old_topk(scores, k)
    t_old = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        new_topk(scores, k)
    t_new = (time.perf_counter() - t0) / reps

    speedup = t_old / t_new
    print(f"n={n:>7}  argsort={t_old*1000:8.4f}ms  argpartition={t_new*1000:8.4f}ms  speedup={speedup:5.2f}x")
