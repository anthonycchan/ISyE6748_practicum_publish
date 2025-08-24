# MARS TENSOR DATASET — GPU-stable CP/Tucker + RX + CP+OCSVM (with fallbacks)
# -----------------------------------------------------------------------------
# - Uses TensorLy + PyTorch (CUDA) for CP/Tucker (float64 on GPU for stability)
# - Train-only per-band standardization
# - Validation (typical-only) for unsupervised selection:
#     * CP/Tucker: choose rank by MIN median reconstruction error on VAL typicals
#     * CP+OCSVM: choose (rank, gamma, nu) by MAX median decision_function on VAL typicals
# - Test (typical+novel): report ROC AUC using continuous scores
# - Robustness: contiguous tensors, HOSVD fallback, CPU (NumPy) fallback per-tile, NaN filtering
# -----------------------------------------------------------------------------

import os
import time
import random
import numpy as np

import tensorly as tl
from tensorly.decomposition import parafac, tucker

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid

# -----------------------------
# Config (update paths as needed)
# -----------------------------
train_data        = "Data/Reduced/Lean/train"        # typical only
validation_data   = "Data/Reduced/Lean/validation"   # typical only
test_typical_data = "Data/Reduced/Lean/test_typical" # typical
test_anomaly_data = "Data/Reduced/Lean/test_novel"   # novel

# Which methods to run
RUN_RX_PIXEL           = True
RUN_CP_RECON_ERROR     = True
RUN_TUCKER_RECON_ERROR = True
RUN_CP_OCSVM           = True

# Search spaces
CP_RANKS          = [8, 12, 16, 20]
TUCKER_RANKS_LIST = [(24, 24, 3), (32, 32, 4), (40, 40, 4)]
OCSVM_PARAM_GRID  = {
    "kernel": ["rbf"],
    "gamma":  list(np.logspace(-6, 1, 16)),  # 1e-6 … 10
    "nu":     [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# -----------------------------
# Backend / Device selection
# -----------------------------
BACKEND = "pytorch"   # GPU preferred; set to "numpy" to force CPU
DEVICE  = "cpu"

def _set_backend():
    global DEVICE
    if BACKEND == "pytorch":
        try:
            import torch
            if torch.cuda.is_available():
                tl.set_backend("pytorch")
                DEVICE = "cuda"
                torch.set_num_threads(1)
                print("[tensorly] Backend: pytorch (CUDA)")
            else:
                tl.set_backend("pytorch")
                DEVICE = "cpu"
                print("[tensorly] Backend: pytorch (CPU fallback)")
        except Exception as e:
            print(f"[tensorly] PyTorch backend not available ({e}); falling back to numpy.")
            tl.set_backend("numpy")
            DEVICE = "cpu"
            print("[tensorly] Backend: numpy")
    else:
        tl.set_backend("numpy")
        DEVICE = "cpu"
        print("[tensorly] Backend: numpy")

_set_backend()

# -----------------------------
# I/O helpers (expects 64x64x6 .npy)
# -----------------------------
def _load_folder(folder, label):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    X = np.empty((len(files), 64, 64, 6), dtype=np.float32)
    y = np.full((len(files),), label, dtype=np.int32)
    for i, f in enumerate(files):
        arr = np.load(os.path.join(folder, f)).astype(np.float32) / 255.0
        X[i] = arr
    return X, y

def read_train(folder):        # typical only
    return _load_folder(folder, label=1)

def read_validation(folder):   # typical only
    return _load_folder(folder, label=1)

def read_test(typ_dir, ano_dir):  # typical + anomaly
    X_typ, y_typ = _load_folder(typ_dir, 1)
    X_ano, y_ano = _load_folder(ano_dir, -1)
    X = np.vstack([X_typ, X_ano])
    y = np.concatenate([y_typ, y_ano])
    return X, y

# -----------------------------
# Standardization (train-only)
# -----------------------------
def fit_band_standardizer(X_train):
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_band_standardizer(X, mu, sigma):
    Z = (X - mu) / sigma
    return np.clip(Z, -20.0, 20.0)

# -----------------------------
# Sanity & cleaning
# -----------------------------
def sanity_report(X, name):
    print(f"[sanity] {name}: N={X.shape[0]}, NaNs={np.isnan(X).any()}, Infs={np.isinf(X).any()}")
    if X.size:
        print(f"[sanity] {name}: min={np.nanmin(X):.6f}, max={np.nanmax(X):.6f}")

def clean_tiles(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -50.0, 50.0)

# -----------------------------
# AUC (explicit anomaly polarity)
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
# Backend helpers (float64 on GPU)
# -----------------------------
def to_backend(x, use_float64=False):
    arr = np.asarray(x, dtype=np.float64 if use_float64 else np.float32, order="C")
    if tl.get_backend() == "pytorch":
        import torch
        t = torch.from_numpy(arr).to(DEVICE, non_blocking=True)
        return t.contiguous()
    else:
        return tl.tensor(arr)

def to_numpy(x):
    if tl.get_backend() == "pytorch":
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return tl.to_numpy(x)

# -----------------------------
# Decompositions (ALS in float64 on GPU)
# -----------------------------
def decompose_tucker_backend(tile_std, ranks):
    Xb = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch"))
    core, factors = tucker(
        Xb, ranks=ranks, init="svd", n_iter_max=500, tol=1e-6
    )
    return core, factors

def decompose_cp_backend(tile_std, rank, random_state=RANDOM_STATE):
    Xb = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch"))
    weights, factors = parafac(
        Xb, rank=rank, init="svd", n_iter_max=500, tol=1e-6,
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
# RX(pixel) baseline (CPU)
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
    md2 = np.einsum("ij,jk,ik->i", dif, Sigma_inv, dif)
    return float(np.nanmean(md2))

def run_rx_baseline(X_train_std, X_test_std, y_test):
    mu_rx, Sigma_inv_rx = fit_rx_background(X_train_std)
    s_test = np.array([rx_score_image(x, mu_rx, Sigma_inv_rx) for x in X_test_std], dtype=float)
    auc = manual_auc(y_test, s_test, positive_label=-1)
    print(f"[RX(pixel)] FINAL AUC={auc:.3f}")
    return auc

# -----------------------------
# Robust reconstruction-error scoring (with fallbacks)
# -----------------------------
def tucker_recon_error(tile_std, ranks):
    # 1) ALS on current backend
    try:
        core, facs = decompose_tucker_backend(tile_std, ranks)
        Xhat_b  = tucker_to_tensor_backend(core, facs)
        resid_b = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch")) - Xhat_b
        resid   = to_numpy(resid_b)
        if np.all(np.isfinite(resid)):
            return float(np.linalg.norm(resid))
    except Exception:
        pass
    # 2) HOSVD (single pass)
    try:
        Xb = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch"))
        core, facs = tucker(Xb, ranks=ranks, init="svd", n_iter_max=1, tol=0)
        Xhat_b  = tucker_to_tensor_backend(core, facs)
        resid_b = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch")) - Xhat_b
        resid   = to_numpy(resid_b)
        if np.all(np.isfinite(resid)):
            return float(np.linalg.norm(resid))
    except Exception:
        pass
    # 3) Fallback to CPU/NumPy for this tile
    if tl.get_backend() == "pytorch":
        import tensorly as tl_local
        old = tl_local.get_backend()
        try:
            tl_local.set_backend("numpy")
            core, facs = tucker(tile_std.astype(np.float32), ranks=ranks, init="svd", n_iter_max=500, tol=1e-6)
            Xhat  = tl_local.tucker_to_tensor((core, facs))
            resid = tile_std - Xhat
            if np.all(np.isfinite(resid)):
                return float(np.linalg.norm(resid))
        except Exception:
            pass
        finally:
            tl_local.set_backend(old)
    return np.nan

def cp_recon_error(tile_std, rank):
    # 1) ALS on current backend
    try:
        w, facs = decompose_cp_backend(tile_std, rank)
        Xhat_b  = cp_to_tensor_backend(w, facs)
        resid_b = to_backend(tile_std, use_float64=(tl.get_backend()=="pytorch")) - Xhat_b
        resid   = to_numpy(resid_b)
        if np.all(np.isfinite(resid)):
            return float(np.linalg.norm(resid))
    except Exception:
        pass
    # 2) Fallback to CPU/NumPy for this tile
    if tl.get_backend() == "pytorch":
        import tensorly as tl_local
        old = tl_local.get_backend()
        try:
            tl_local.set_backend("numpy")
            w, facs = parafac(tile_std.astype(np.float32), rank=rank, init="svd", n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=RANDOM_STATE)
            Xhat  = tl_local.cp_to_tensor((w, facs))
            resid = tile_std - Xhat
            if np.all(np.isfinite(resid)):
                return float(np.linalg.norm(resid))
        except Exception:
            pass
        finally:
            tl_local.set_backend(old)
    return np.nan

def select_rank_by_val_error(method_name, scorer, rank_space, X_val_std):
    """Unsupervised: choose rank minimizing median reconstruction error on VAL typicals."""
    if X_val_std.shape[0] == 0:
        print(f"[{method_name}] VAL set is empty — cannot select rank.")
        return None
    best_rank, best_stat = None, np.inf
    for r in rank_space:
        errs = np.array([scorer(x, r) for x in X_val_std], dtype=float)
        errs = errs[np.isfinite(errs)]
        if errs.size == 0:
            print(f"[{method_name}] rank={r} -> all NaN, skipping.")
            continue
        med = float(np.median(errs))
        print(f"[{method_name}] rank={r} VAL median error={med:.6f} (from {errs.size} tiles)")
        if med < best_stat:
            best_stat, best_rank = med, r
    if best_rank is None:
        print(f"[{method_name}] No valid rank (all NaN or empty).")
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
# CP + OC-SVM (unsupervised on VAL typicals)
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
            feats.append(None); continue
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
    if not feats:
        return None, None
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

        Xtr, _    = stack_filter_features(feats_train)
        Xv, idx_v = stack_filter_features(feats_val)
        Xt, idx_t = stack_filter_features(feats_test)

        if Xtr is None or Xv is None or Xt is None:
            print(f"[CP+OCSVM] rank={r}: insufficient features (None/NaN). Skipping.")
            continue

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xv_s  = scaler.transform(Xv)
        Xt_s  = scaler.transform(Xt)

        if use_pca_whiten:
            pca   = PCA(whiten=True, svd_solver="auto", random_state=RANDOM_STATE)
            Xtr_w = pca.fit_transform(Xtr_s)
            Xv_w  = pca.transform(Xv_s)
            Xt_w  = pca.transform(Xt_s)
        else:
            pca   = None
            Xtr_w, Xv_w, Xt_w = Xtr_s, Xv_s, Xt_s

        # Unsupervised selection on VAL typicals (maximize median decision_function)
        best_metric_r, best_params_r, best_model_r = -np.inf, None, None
        for params in ParameterGrid(param_grid):
            try:
                model = OneClassSVM(**params).fit(Xtr_w)
                s_val = model.decision_function(Xv_w).ravel()  # higher = more inlier-ish
                if not np.any(np.isfinite(s_val)):
                    continue
                med = float(np.nanmedian(s_val))
                inlier_rate = float((s_val >= 0).mean())
                metric = med + 0.01 * inlier_rate  # tiny tie-break
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
    s_test = -model.decision_function(Xt_w).ravel()   # anomaly score = -decision_function
    auc = manual_auc(y_test_f, s_test, positive_label=-1)
    print(f"[CP+OCSVM] SELECTED rank={best_rank} params={best_params} | FINAL AUC={auc:.3f}")
    return best_rank, best_params, auc

# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    print("Loading data ...")
    X_train, _      = read_train(train_data)           # typical-only
    X_val, _        = read_validation(validation_data) # typical-only
    X_test, y_test  = read_test(test_typical_data, test_anomaly_data)

    # Per-band scaler on TRAIN only
    mu_b, sigma_b = fit_band_standardizer(X_train)
    X_train_std = apply_band_standardizer(X_train, mu_b, sigma_b)
    X_val_std   = apply_band_standardizer(X_val,   mu_b, sigma_b)
    X_test_std  = apply_band_standardizer(X_test,  mu_b, sigma_b)

    print(f"X_val {X_val_std.shape[0]}")
    print(f"Train typicals: {X_train.shape[0]} | VAL (typical): {X_val.shape[0]} | TEST: {X_test.shape[0]}")
    print(f"TEST class balance: {(y_test == -1).sum()} anomalies / {len(y_test)}")

    # Sanity checks + auto-clean if needed
    sanity_report(X_val_std,  "VAL std")
    sanity_report(X_test_std, "TEST std")
    if np.isnan(X_val_std).any() or np.isinf(X_val_std).any():
        print("[sanity] Cleaning VAL tiles ...")
        X_val_std = clean_tiles(X_val_std)
    if np.isnan(X_test_std).any() or np.isinf(X_test_std).any():
        print("[sanity] Cleaning TEST tiles ...")
        X_test_std = clean_tiles(X_test_std)

    # 1) RX baseline (CPU)
    if RUN_RX_PIXEL:
        run_rx_baseline(X_train_std, X_test_std, y_test)
        print()

    # 2) CP reconstruction-error (GPU if available, with fallbacks)
    if RUN_CP_RECON_ERROR:
        best_cp_rank = select_rank_by_val_error("CP recon-error", cp_recon_error, CP_RANKS, X_val_std)
        eval_recon_error_on_test("CP recon-error", cp_recon_error, best_cp_rank, X_test_std, y_test)
        print()

    # 3) Tucker reconstruction-error (GPU if available, with fallbacks)
    if RUN_TUCKER_RECON_ERROR:
        best_tucker_rank = select_rank_by_val_error("Tucker recon-error", tucker_recon_error, TUCKER_RANKS_LIST, X_val_std)
        eval_recon_error_on_test("Tucker recon-error", tucker_recon_error, best_tucker_rank, X_test_std, y_test)
        print()

    # 4) CP + OC-SVM (features from GPU CP; SVM/PCA on CPU)
    if RUN_CP_OCSVM:
        run_cp_ocs_unsup(
            X_train_std, X_val_std, X_test_std, y_test,
            ranks=CP_RANKS,
            param_grid=OCSVM_PARAM_GRID,
            use_pca_whiten=True
        )
        print()

    print(f"Total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
