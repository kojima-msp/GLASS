"""Bake fixed randomness into per-index dataset files for NY_Subway and JapanCOVID19.

    index 0-{N_train-1}   -> train signal
    index {N_train}-{N_train+N_TEST-1} -> test  signal
"""

from pathlib import Path

import numpy as np

MASTER_SEED = 20240101
N_TRAIN, N_TEST = 10, 5

ROOT = Path(__file__).resolve().parent  # .../datasets


def load_dataset(subdir):
    """Return (train_signal, test_signal, L, DeltaG, W) for a real dataset."""
    if subdir == "NY_Subway":
        src = np.load(ROOT / subdir / "out_full.npz")
        return src["train"], src["test"], src["L"], src["DeltaG"], src["A"]
    src = np.load(ROOT / subdir / "data.npz")   # JapanCOVID19
    return src["train"], src["test"], src["L"], src["DeltaG"], src["W"]


def preprocess(rng, subdir):
    train, test, L, DeltaG, W = load_dataset(subdir)
    for idx, gt in enumerate([train] * N_TRAIN + [test] * N_TEST):
        np.savez(ROOT / subdir / f"data_{idx:02d}.npz", groundtruth=gt, L=L, DeltaG=DeltaG, W=W,
                 uniform=rng.random(gt.shape),
                 noise=gt.std() * rng.standard_normal(gt.shape))   # scaled by the signal std
    print(f"  {subdir}: {N_TRAIN} train {train.shape} + {N_TEST} test {test.shape}"
          f" -> data_00..data_{N_TRAIN + N_TEST - 1:02d}.npz")


def main():
    print(f"Generating fixed randomness (master seed = {MASTER_SEED})")
    rng = np.random.default_rng(MASTER_SEED)
    preprocess(rng, "NY_Subway")
    preprocess(rng, "JapanCOVID19")
    print("done.")


if __name__ == "__main__":
    main()