from typing import List, Tuple, Iterable
import numpy as np

def split_ids(
    start: int = 123,
    end: int = 226,
    exclude: Iterable[int] = (200, 207, 208, 220, 221, 224, 225),
    train_frac: float = 0.80,
    seed: int | None = 42,  # fix for reproducibility; set to None for fresh randomness
) -> Tuple[List[int], List[int]]:
    """
    Returns (train_ids, test_ids) as sorted lists with:
      - numbers from `start`..`end` inclusive, excluding `exclude`
      - ~80/20 split using nearest-integer rounding (most accurate to target ratio)
      - no overlap, full coverage.
    """
    # 1) Build the universe, apply exclusions
    all_ids = list(range(start, end + 1))
    exclude_set = set(exclude)
    pool = sorted([x for x in all_ids if x not in exclude_set])

    # 2) Decide sizes using *nearest* integer to target proportion
    n = len(pool)
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0,1).")
    n_test = int(round((1 - train_frac) * n))
    n_test = max(1, min(n - 1, n_test))  # keep both splits non-empty
    n_train = n - n_test

    # 3) Sample without replacement using a modern RNG
    rng = np.random.default_rng(seed)
    test_ids = sorted(rng.choice(pool, size=n_test, replace=False).tolist())
    train_ids = sorted(sorted(set(pool) - set(test_ids)))

    # 4) Safety checks (no overlap, full coverage, sizes match)
    assert set(train_ids).isdisjoint(test_ids), "Overlap detected between TRAIN and TEST!"
    assert len(train_ids) == n_train and len(test_ids) == n_test, "Split sizes don't match plan."
    assert set(train_ids).union(test_ids) == set(pool), "Not all eligible IDs were assigned."

    return train_ids, test_ids


if __name__ == "__main__":
    train, test = split_ids()
    print(f"Total eligible: {len(train) + len(test)}")
    print(f"TRAIN ({len(train)} ids): {train}")
    print(f"TEST  ({len(test)} ids): {test}")

    # Optional: save to disk
    # with open("train_ids.txt", "w") as f: f.write("\n".join(map(str, train)))
    # with open("test_ids.txt", "w") as f: f.write("\n".join(map(str, test)))
