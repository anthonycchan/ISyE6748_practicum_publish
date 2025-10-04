# MARS TENSOR DATASET — Unsupervised VAL (typical-only) selection + FINAL test AUC
# ------------------------------------------------------------------------------
# Uses:
#   - Train typicals: fit per-band scaler (mu, sigma)
#   - Validation typicals: unsupervised selection
#        * CP/Tucker: pick rank that MINIMIZES median reconstruction error on VAL typicals
#        * CP+OC-SVM: pick (rank, gamma, nu) that MAXIMIZES median decision_function on VAL typicals
#   - Test (typical + novel): report ROC AUC using continuous scores
#
# Methods:
#   - RX(pixel) baseline (no hyperparams)
#   - CP reconstruction-error
#   - Tucker reconstruction-error
#   - CP factors + One-Class SVM
#
# Backend:
#   - Defaults to 'numpy' for stability. Set BACKEND = "pytorch" after verifying correctness.

import os
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorly as tl
from tensorly.decomposition import parafac, tucker

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score  # used only for optional checks
# manual AUC implemented below to keep label polarity explicit

# -----------------------------
# Config: update paths per your message
# -----------------------------
train_data       = "Data/Reduced/Lean/train"        # typical only
validation_data  = "Data/Reduced/Lean/validation"   # typical only
test_typical_data= "Data/Reduced/Lean/test_typical" # typical
test_anomaly_data= "Data/Reduced/Lean/test_novel"   # novel/anomaly

# Methods to run
RUN_RX_PIXEL              = True
RUN_CP_RECON_ERROR        = True
RUN_TUCKER_RECON_ERROR    = True
RUN_CP_OCSVM              = True

# Search spaces
CP_RANKS          = [8, 12, 16, 20]
TUCKER_RANKS_LIST = [(24, 24, 3), (32, 32, 4), (40, 40, 4)]

OCSVM_PARAM_GRID = {
    'kernel': ['rbf'],
    'gamma':  list(np.logspace(-6, 1, 16)),  # 1e-6 … 10
    'nu':     [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
}

RANDOM_STATE = 42
N_WORKERS    = max(1, (os.cpu_count() or 2) - 1)

# Backend: start with numpy; switch to pytorch once you confirm all AUCs look sane
BACKEND = "numpy"  # change to "pytorch" when ready

def _set_backend():
    if BACKEND == "pytorch":
        try:
            import torch  # noqa: F401
            tl.set_backend('pytorch')
            print("[tensorly] Backend: pytorch")
            return
        except Exception as e:
            print(f"[tensorly] pytorch backend unavailable ({e}); using numpy.")
    tl.set_backend('numpy')
    print("[tensorly] Backend: numpy")

_set_backend()

# -----------------------------
# IO helpers (expects 64x64x6 .npy tiles)
# -----------------------------
def _load_folder(folder, label):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    X = np.empty((len(files), 64, 64, 6), dtype=np.float32)
    y = np.empty((len(files),), dtype=np.int32)
    for i, f in enumerate(files):
        arr = np.load(os.path.join(folder, f)).astype(np.float32) / 255.0
        X[i] = arr
        y[i] = label
    return X, y

def read_train(directory):        # typical only
    return _load_folder(directory, label=1)

def read_validation(directory):   # typical only
    return _load_folder(directory, label=1)

def read_test(typ_dir, ano_dir):  # typical + anomaly
    X_typ, y_typ = _load_folder(typ_dir, 1)
    X_ano, y_ano = _load_folder(ano_dir, -1)
    X = np.vstack([X_typ, X_ano])
    y = np.concatenate([y_typ, y_ano])
    return X, y

# -----------------------------
# Standardization
# -----------------------------
def fit_band_standardizer(X_train):
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_band_standardizer(X, mu, sigma):
    Z = (X - mu) / sigma
    # guard against extreme z-scores
    return np.clip(Z, -20.0, 20.0)

# -----------------------------
# AUC util (explicit anomaly polarity)
# -----------------------------
def manual_auc(y_true, scores, positive_label=-1):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    pos = (y_true == positive_label)
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0 or scores.size == 0:
        return float("nan")
    pos_scores = scores[pos]
    neg_scores = scores[~pos]
    greater = (pos_scores[:, None] >  neg_scores[None, :]).sum()
    ties    = (pos_scores[:, None] == neg_scores[None, :]).sum()
    return float((greater + 0.5 * ties) / (n_pos * n_neg))

# -----------------------------
# Backend helpers
# -----------------------------
def to_backend(x):
    return tl.tensor(np.asarray(x, dtype=np.float32))

def to_numpy(x):
    try:
        return tl.to_numpy(x)
    except Exception:
        return np.array(x)

# -----------------------------
# Decompositions
# -----------------------------
def decompose_tucker_backend(tile_std, ranks):
    Xb = to_backend(tile_std)
    core, factors = tucker(Xb, ranks=ranks, init='svd', n_iter_max=500, tol=1e-6)
    return core, factors

def decompose_cp_backend(tile_std, rank, random_state=RANDOM_STATE):
    Xb = to_backend(tile_std)
    weights, factors = parafac(
        Xb, rank=rank, init='svd', n_iter_max=500, tol=1e-6,
        normalize_factors=True, random_state=random_state
    )
    return weights, factors

def cp_to_tensor_backend(weights, factors):
    return tl.cp_to_tensor((weights, factors))

def tucker_to_tensor_backend(core, factors):
    return tl.tucker_to_tensor((core, factors))

# -----------------------------
# Features for OC-SVM
# -----------------------------
def features_from_cp_backend(weights, factors):
    w = to_numpy(weights).ravel().astype(np.float32)
    parts = [w]
    for Fm in factors:
        parts.append(to_numpy(Fm).ravel().astype(np.float32))
    return np.concatenate(parts, axis=0)

# -----------------------------
# RX(pixel) baseline
# -----------------------------
def fit_rx_background(X_train_std):
    pixels = X_train_std.reshape(-1, X_train_std.shape[-1])  # (N*4096, 6)
    mu = pixels.mean(axis=0)
    Sigma = np.cov(pixels, rowvar=False)
    Sigma_inv = np.linalg.pinv(Sigma)
    return mu.astype(np.float64), Sigma_inv.astype(np.float64)

def rx_score_image(img_std, mu, Sigma_inv):
    P = img_std.reshape(-1, img_std.shape[-1])
    dif = P - mu
    md2 = np.einsum('ij,jk,ik->i', dif, Sigma_inv, dif)  # Mahalanobis^2
    return float(np.nanmean(md2))

def run_rx_baseline(X_train_std, X_test_std, y_test):
    mu_rx, Sigma_inv_rx = fit_rx_background(X_train_std)
    s_test = np.array([rx_score_image(x, mu_rx, Sigma_inv_rx) for x in X_test_std], dtype=float)
    auc = manual_auc(y_test, s_test, positive_label=-1)
    print(f"[RX(pixel)] FINAL AUC={auc:.3f}")
    return auc

# -----------------------------
# Reconstruction-error scoring
# -----------------------------
def cp_recon_error(tile_std, rank):
    try:
        w, facs = decompose_cp_backend(tile_std, rank)
        Xhat_b = cp_to_tensor_backend(w, facs)
        resid_b = to_backend(tile_std) - Xhat_b
        resid = to_numpy(resid_b)
        if not np.all(np.isfinite(resid)): return np.nan
        return float(np.linalg.norm(resid))
    except Exception:
        return np.nan

def tucker_recon_error(tile_std, ranks):
    try:
        core, facs = decompose_tucker_backend(tile_std, ranks)
        Xhat_b = tucker_to_tensor_backend(core, facs)
        resid_b = to_backend(tile_std) - Xhat_b
        resid = to_numpy(resid_b)
        if not np.all(np.isfinite(resid)): return np.nan
        return float(np.linalg.norm(resid))
    except Exception:
        return np.nan

def select_rank_by_val_error(method_name, scorer, rank_space, X_val_std):
    """Unsupervised: choose rank minimizing median reconstruction error on VAL typicals."""
    best_rank, best_stat = None, np.inf
    for r in rank_space:
        errs = np.array([scorer(x, r) for x in X_val_std], dtype=float)
        med = np.nanmedian(errs)
        print(f"[{method_name}] rank={r} VAL median error={med:.6f}")
        if np.isfinite(med) and med < best_stat:
            best_stat, best_rank = med, r
    if best_rank is None:
        print(f"[{method_name}] No valid rank (all NaN).")
    else:
        print(f"[{method_name}] SELECTED rank={best_rank} (VAL median error={best_stat:.6f})")
    return best_rank

def eval_recon_error_on_test(method_name, scorer, best_rank, X_test_std, y_test):
    if best_rank is None:
        return None
    s_test = np.array([scorer(x, best_rank) for x in X_test_std], dtype=float)
    auc = manual_auc(y_test, s_test, positive_label=-1)
    print(f"[{method_name}] FINAL AUC={auc:.3f}")
    return auc

# -----------------------------
# CP + OC-SVM (unsupervised selection on VAL typicals)
# -----------------------------
def decompose_cp_list_backend(X_std, rank):
    out = []
    for x in X_std:
        try:
            out.append(decompose_cp_backend(x, rank))
        except Exception:
            out.append(None)
    return out

def features_from_cp_list_backend(cp_list):
    feats = []
    for wf in cp_list:
        if wf is None:
            feats.append(None)
            continue
        w, facs = wf
        try:
            feats.append(features_from_cp_backend(w, facs))
        except Exception:
            feats.append(None)
    return feats

def stack_filter_features(feat_list):
    feats, idx = [], []
    for i, f in enumerate(feat_list):
        if f is not None and np.all(np.isfinite(f)):
            feats.append(f); idx.append(i)
    if not feats: return None, None
    X = np.vstack(feats).astype(np.float32)
    return X, np.array(idx, dtype=int)

def run_cp_ocs_unsup(X_train_std, X_val_std, X_test_std, y_test,
                     ranks, param_grid, use_pca_whiten=True):
    best_metric, best_rank, best_params = -np.inf, None, None
    best_payload = None

    for r in ranks:
        print(f"[CP+OCSVM] rank={r}: decomposing TRAIN/VAL/TEST ...")
        cp_train = decompose_cp_list_backend(X_train_std, r)
        cp_val   = decompose_cp_list_backend(X_val_std,   r)
        cp_test  = decompose_cp_list_backend(X_test_std,  r)

        feats_train = features_from_cp_list_backend(cp_train)
        feats_val   = features_from_cp_list_backend(cp_val)
        feats_test  = features_from_cp_list_backend(cp_test)

        Xtr, _ = stack_filter_features(feats_train)
        Xv,  idx_v  = stack_filter_features(feats_val)
        Xt,  idx_t  = stack_filter_features(feats_test)

        if Xtr is None or Xv is None or Xt is None:
            print(f"[CP+OCSVM] rank={r}: insufficient features (None/NaN). Skipping.")
            continue

        # all VAL labels are +1 (typical-only)
        # build pipeline
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xv_s  = scaler.transform(Xv)
        Xt_s  = scaler.transform(Xt)

        if use_pca_whiten:
            pca = PCA(whiten=True, svd_solver='auto', random_state=RANDOM_STATE)
            Xtr_w = pca.fit_transform(Xtr_s)
            Xv_w  = pca.transform(Xv_s)
            Xt_w  = pca.transform(Xt_s)
        else:
            pca = None
            Xtr_w, Xv_w, Xt_w = Xtr_s, Xv_s, Xt_s

        # Unsupervised selection on VAL typicals:
        # Maximize median decision_function (higher => more inlier-ish).
        # Tie-breaker: inlier rate at default threshold 0.
        best_metric_r, best_params_r, best_model_r = -np.inf, None, None
        for params in ParameterGrid(param_grid):
            try:
                model = OneClassSVM(**params).fit(Xtr_w)
                s_val = model.decision_function(Xv_w).ravel()  # + = inlier, - = outlier
                if not np.any(np.isfinite(s_val)):
                    continue
                med = np.nanmedian(s_val)
                inlier_rate = float((s_val >= 0).mean())
                metric = med + 0.01 * inlier_rate  # small tie-break
                if metric > best_metric_r:
                    best_metric_r, best_params_r, best_model_r = metric, params, model
            except Exception:
                continue

        print(f"[CP+OCSVM] rank={r} BEST VAL metric={best_metric_r:.6f} params={best_params_r}")

        if np.isfinite(best_metric_r) and best_metric_r > best_metric:
            best_metric, best_rank, best_params = best_metric_r, r, best_params_r
            best_payload = (scaler, pca, best_model_r, Xt_w, y_test[idx_t])

    if best_payload is None:
        print("[CP+OCSVM] No valid model found.")
        return None, None, None

    scaler, pca, model, Xt_w, y_test_f = best_payload
    # anomaly score = -decision_function (higher => more anomalous)
    s_test = -model.decision_function(Xt_w).ravel()
    auc = manual_auc(y_test_f, s_test, positive_label=-1)
    print(f"[CP+OCSVM] SELECTED rank={best_rank} params={best_params} | FINAL AUC={auc:.3f}")
    return best_rank, best_params, auc

# -----------------------------
# Main
# -----------------------------
def main():
    start = time.time()
    print("Loading data ...")
    X_train, _      = read_train(train_data)           # typical-only
    X_val, _        = read_validation(validation_data) # typical-only
    X_test, y_test  = read_test(test_typical_data, test_anomaly_data)  # typical + novel

    # Fit per-band scaler on TRAIN typicals only
    mu_b, sigma_b = fit_band_standardizer(X_train)
    X_train_std = apply_band_standardizer(X_train, mu_b, sigma_b)
    X_val_std   = apply_band_standardizer(X_val,   mu_b, sigma_b)
    X_test_std  = apply_band_standardizer(X_test,  mu_b, sigma_b)

    print(f"Train typicals: {X_train.shape[0]} | VAL (typical): {X_val.shape[0]} | TEST: {X_test.shape[0]}")
    print(f"TEST class balance: {(y_test==-1).sum()} anomalies / {len(y_test)}")

    # 1) RX(pixel) baseline
    if RUN_RX_PIXEL:
        run_rx_baseline(X_train_std, X_test_std, y_test)
        print()

    # 2) CP reconstruction-error (unsupervised rank selection)
    if RUN_CP_RECON_ERROR:
        best_cp_rank = select_rank_by_val_error("CP recon-error", cp_recon_error, CP_RANKS, X_val_std)
        eval_recon_error_on_test("CP recon-error", cp_recon_error, best_cp_rank, X_test_std, y_test)
        print()

    # 3) Tucker reconstruction-error (unsupervised rank selection)
    if RUN_TUCKER_RECON_ERROR:
        best_tucker_rank = select_rank_by_val_error("Tucker recon-error", tucker_recon_error, TUCKER_RANKS_LIST, X_val_std)
        eval_recon_error_on_test("Tucker recon-error", tucker_recon_error, best_tucker_rank, X_test_std, y_test)
        print()

    # 4) CP factors + OC-SVM (unsupervised selection on VAL typicals)
    if RUN_CP_OCSVM:
        run_cp_ocs_unsup(
            X_train_std, X_val_std, X_test_std, y_test,
            ranks=CP_RANKS,
            param_grid=OCSVM_PARAM_GRID,
            use_pca_whiten=True
        )
        print()

    print(f"Total runtime: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
