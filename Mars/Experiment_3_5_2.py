#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce RX(pixel) and PCA(flat) baselines for Mastcam novelty detection
with settings aligned to the paper's reported AUCs:

  - RX(pixel): background μ, Σ learned from TRAIN pixels.
               Image score = MEAN of per-pixel RX scores.
               Covariance uses shrinkage (Ledoit–Wolf by default) on a large
               random sample of train pixels for stability (recommended).

  - PCA(flat): PCA fit on FLATTENED TRAIN images (very low k ~ 5).
               Novelty score = L2 reconstruction error per image.

Expected overall AUCs (paper, combined test set):
  RX(pixel) ~ 0.72
  PCA(flat) ~ 0.50

Paths (fixed as requested):
  train_data        = "Data/Full/train_typical"
  validation_data   = "Data/Full/validation_typical"   # not used here
  test_typical_data = "Data/Full/test_typical"
  test_anomaly_data = "Data/Full/test_novel/all"

Usage examples:
  python mastcam_rx_pca_repro.py --rx_cov ledoitwolf --rx_sample_pixels 1500000 --pca_k 5
  python mastcam_rx_pca_repro.py --rx_cov oas --rx_sample_pixels 1000000 --pca_k 5
  python mastcam_rx_pca_repro.py --rx_cov empirical --pca_k 5   # empirical as a check

Dependencies:
  numpy, scikit-learn
"""

import os
import time
import argparse
import numpy as np
from typing import Tuple

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance

# ---------- Fixed Paths ----------
train_data        = "Data/Full/train_typical"
validation_data   = "Data/Full/validation_typical"  # not used here
test_typical_data = "Data/Full/test_typical"
test_anomaly_data = "Data/Full/test_novel/all"

# ---------- I/O ----------
def _list_npy(dirpath):
    return sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])

def read_dir_as_array(dirpath: str) -> np.ndarray:
    """
    Load .npy tiles from a directory into a single array of shape (N, H, W, C),
    normalized to [0,1] if they look like 0..255 data.
    """
    files = _list_npy(dirpath)
    if not files:
        raise FileNotFoundError(f"No .npy files found in {dirpath}")
    # Read first to infer shape
    first = np.load(os.path.join(dirpath, files[0]))
    if first.ndim == 2:
        H, W = first.shape
        C = 1
    elif first.ndim == 3:
        H, W, C = first.shape
    else:
        raise ValueError(f"Unexpected array shape {first.shape} in {files[0]}")

    X = np.empty((len(files), H, W, C), dtype=np.float32)
    for i, f in enumerate(files):
        arr = np.load(os.path.join(dirpath, f)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        # Normalize to [0,1] if values look like 0..255
        if arr.max() > 1.5:
            arr = arr / 255.0
        X[i] = arr
    return X

def read_test_pair(typ_dir: str, ano_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X_typ = read_dir_as_array(typ_dir)
    X_ano = read_dir_as_array(ano_dir)
    X = np.concatenate([X_typ, X_ano], axis=0)
    y = np.concatenate([
        np.zeros(len(X_typ), dtype=np.int32),
        np.ones(len(X_ano), dtype=np.int32)
    ], axis=0)  # 1 = anomaly
    return X, y

# ---------- RX(pixel) ----------
def sample_train_pixels(X_train: np.ndarray, max_pixels: int, seed: int = 123) -> np.ndarray:
    """
    Return a random sample of up to max_pixels rows of shape (C,),
    where each row is one pixel spectrum from the train tiles.
    """
    rng = np.random.RandomState(seed)
    N, H, W, C = X_train.shape
    total = N * H * W
    take = min(max_pixels, total)
    # sample flat indices without replacement
    idx = rng.choice(total, size=take, replace=False)
    # unravel to (n,2) for pixel positions within tiles
    # Map flat index to (n_tile, yx) is more involved; easiest is to reshape
    Xv = X_train.reshape(-1, C)
    sample = Xv[idx].astype(np.float64)
    return sample

def fit_rx_background_shrinkage(X_train: np.ndarray,
                                cov_type: str = "ledoitwolf",
                                sample_pixels: int = 1_500_000,
                                seed: int = 123):
    """
    Fit RX background using shrinkage covariance on a large random pixel sample.
    Returns:
      mu:  (C,)   background mean
      inv: (C,C)  inverse covariance
    """
    # Draw a big random pixel sample for stable covariance estimation
    Xs = sample_train_pixels(X_train, max_pixels=sample_pixels, seed=seed)  # (M, C)
    if cov_type.lower() == "ledoitwolf":
        est = LedoitWolf().fit(Xs)
    elif cov_type.lower() == "oas":
        est = OAS().fit(Xs)
    elif cov_type.lower() == "empirical":
        est = EmpiricalCovariance().fit(Xs)
    else:
        raise ValueError("cov_type must be one of: ledoitwolf, oas, empirical")

    mu = np.asarray(est.location_, dtype=np.float64)        # (C,)
    cov = np.asarray(est.covariance_, dtype=np.float64)     # (C,C)
    # Numerical safety
    cov = 0.5 * (cov + cov.T)
    # Try inverse; fallback to pseudo-inverse
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov, rcond=1e-12)
    return mu, inv

def rx_pixel_scores_mean(X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """
    RX(pixel) per-image scores: mean RX across pixels.
    X: (N, H, W, C); mu: (C,), inv_cov: (C,C)
    Returns: scores (N,) where larger means "more anomalous".
    """
    N = X.shape[0]
    out = np.empty((N,), dtype=np.float64)
    for i in range(N):
        F = X[i].reshape(-1, X.shape[3]).astype(np.float64)  # (H*W, C)
        D = F - mu[None, :]                                  # center
        # RX per pixel: (x-mu)^T invΣ (x-mu)
        q = np.einsum('ij,jk,ik->i', D, inv_cov, D, optimize=True)
        out[i] = float(q.mean())  # paper: mean across pixels
    return out

def run_rx_pixel(train_dir: str, test_typ_dir: str, test_ano_dir: str,
                 cov_type: str = "ledoitwolf",
                 sample_pixels: int = 1_500_000,
                 seed: int = 123) -> float:
    print("\n[RX] Fitting RX background (typical-only train)...")
    t0 = time.time()
    Xtr = read_dir_as_array(train_dir)
    mu, inv_cov = fit_rx_background_shrinkage(
        Xtr, cov_type=cov_type, sample_pixels=sample_pixels, seed=seed
    )
    print(f"[RX] Fit in {time.time()-t0:.2f}s  | cov={cov_type}, sample_pixels={sample_pixels}")

    print("[RX] Scoring test tiles ...")
    Xt, yt = read_test_pair(test_typ_dir, test_ano_dir)
    t1 = time.time()
    scores = rx_pixel_scores_mean(Xt, mu, inv_cov)  # higher => more anomalous
    auc = roc_auc_score(yt, scores)                 # yt: 1 = anomaly
    print(f"[RX] AUC (anomaly-positive): {auc:.4f}  (agg=mean) | time={time.time()-t1:.2f}s")
    return auc

# ---------- PCA(flat) ----------
def flatten_tiles(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1)).astype(np.float32)

def pca_flat_recon_scores(train_dir: str, test_typ_dir: str, test_ano_dir: str,
                          n_components: int = 5,
                          use_incremental: bool = False,
                          batch_size: int = 64) -> float:
    """
    PCA on flattened TRAIN images with very small k (default 5),
    novelty = L2 reconstruction error per image.
    """
    print("\n[PCA] Fitting PCA on flattened TRAIN images ...")
    t0 = time.time()
    Xtr = read_dir_as_array(train_dir)
    Xtr_flat = flatten_tiles(Xtr)

    if use_incremental:
        pca = IncrementalPCA(n_components=n_components)
        for s in range(0, Xtr_flat.shape[0], batch_size):
            pca.partial_fit(Xtr_flat[s:s+batch_size])
    else:
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0)
        pca.fit(Xtr_flat)

    print(f"[PCA] Fit in {time.time()-t0:.2f}s | k={n_components} (whiten=False)")

    print("[PCA] Scoring test tiles ...")
    Xt, yt = read_test_pair(test_typ_dir, test_ano_dir)
    Xt_flat = flatten_tiles(Xt)
    Z = pca.transform(Xt_flat)
    Xhat = pca.inverse_transform(Z)
    err = np.sum((Xt_flat - Xhat) ** 2, axis=1).astype(np.float64)  # higher => more anomalous
    auc = roc_auc_score(yt, err)
    print(f"[PCA] AUC (anomaly-positive): {auc:.4f}")
    return auc

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Mastcam RX(pixel) & PCA(flat) reproduction")
    ap.add_argument("--rx_cov", type=str, default="ledoitwolf",
                    choices=["ledoitwolf", "oas", "empirical"],
                    help="Covariance estimator for RX background (shrinkage recommended).")
    ap.add_argument("--rx_sample_pixels", type=int, default=1_500_000,
                    help="How many TRAIN pixels to sample for RX covariance fit.")
    ap.add_argument("--rx_seed", type=int, default=123)

    ap.add_argument("--pca_k", type=int, default=5,
                    help="Number of PCs for PCA(flat). Very small k (~5) matches paper's ~0.50 AUC.")
    ap.add_argument("--pca_incremental", action="store_true",
                    help="Use IncrementalPCA for memory-constrained setups.")
    args = ap.parse_args()

    # Sanity on required dirs
    for d in [train_data, test_typical_data, test_anomaly_data]:
        if not (os.path.isdir(d) and _list_npy(d)):
            raise FileNotFoundError(f"Missing/empty directory: {d}")

    # RX(pixel)
    auc_rx = run_rx_pixel(
        train_data, test_typical_data, test_anomaly_data,
        cov_type=args.rx_cov, sample_pixels=args.rx_sample_pixels, seed=args.rx_seed
    )

    # PCA(flat)
    auc_pca = pca_flat_recon_scores(
        train_data, test_typical_data, test_anomaly_data,
        n_components=args.pca_k, use_incremental=args.pca_incremental
    )

    print("\n=== Summary (paper targets) ===")
    print(f"RX(pixel) AUC = {auc_rx:.4f}   (paper ~ 0.72)")
    print(f"PCA(flat)  AUC = {auc_pca:.4f} (paper ~ 0.50)")

if __name__ == "__main__":
    main()
