# Imports
import os
import time
import random
import torch
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed

# TensorLy
import tensorly as tl
from tensorly.decomposition import parafac as _tl_parafac, tucker as _tl_tucker

# ML utils
from sklearn import metrics
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest

# Keras (for autoencoder)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Memory logging
from utils_mem import peak_ram

random.seed(1)

# Paths & toggles
#train_data        = "Data/Full/train_typical"        # typical only
#validation_data   = "Data/Full/validation_typical"   # typical only
#test_typical_data = "Data/Full/test_typical" # typical
#test_anomaly_data = "Data/Full/test_novel/all"   # novel

train_data        = "../../Data/Reduced/set_1/train"        # typical only
validation_data   = "../../Data/Reduced/set_1/validation"   # typical only
test_typical_data = "../../Data/Reduced/set_1/test_typical_200" # typical
#test_anomaly_data = "Data/Reduced/set_1/test_novel"   # novel
test_anomaly_data = "../../Data/Full/test_novel/bedrock"   # novel
#test_anomaly_data = "Data/Full/test_novel/broken-rock"   # novel
#test_anomaly_data = "Data/Full/test_novel/drill-hole"   # novel
#test_anomaly_data = "Data/Full/test_novel/drt"   # novel
#test_anomaly_data = "Data/Full/test_novel/dump-pile"   # novel
#test_anomaly_data = "Data/Full/test_novel/float"   # novel
#test_anomaly_data = "Data/Full/test_novel/meteorite"   # novel
#test_anomaly_data = "Data/Full/test_novel/scuff"   # novel
#test_anomaly_data = "Data/Full/test_novel/veins"   # novel

use_predefined_rank = True
enable_tucker_oc_svm = True
enable_tucker_autoencoder = True
enable_tucker_isolation_forest = True
enable_cp_oc_svm = True
enable_cp_autoencoder = True
enable_cp_isolation_forest = True
enable_pca_oc_svm = True
enable_pca_autoencoder = True
enable_pca_isolation_forest = True

no_decomposition = False
RUN_CP_VISUALIZATION = False
RUN_TUCKER_VISUALIZATION = False

TRAINING_EVALUATION = True

# Optional: standardize bands using TRAIN stats
USE_BAND_STANDARDIZE = True

# Dataset reduction controls
REDUCE_DATASETS = True
REDUCE_TRAIN_N = 1500
REDUCE_VAL_N = math.ceil(REDUCE_TRAIN_N * 0.20)     # 20% of training data
REDUCE_TEST_TYP_N = 44
REDUCE_TEST_ANO_N = 11
REDUCE_SEED = 1
VAL_FRACTION = 0.5  # only used if no separate validation dir


# TensorLy backend + device toggles
TL_BACKEND = "pytorch"   # change to "numpy" to force CPU
#TL_BACKEND = "numpy"
DEVICE = "cuda"
USE_GPU_CP = True

# Options: "both" (core + factors), "core" (core only), "factors" (factors only)
TUCKER_FEATURE_MODE = "core"

# --- IF + typical-only VAL controls ---
USE_VAL_FOR_IF = True       # use VAL (typical-only) for model selection + threshold
VAL_FP_TARGET  = 0.05       # desired false-positive rate on typical VAL (e.g., 0.05 -> 95th percentile)


def _set_tl_backend():
    """Choose TensorLy backend. If PyTorch is selected, prefer CUDA when available."""
    global DEVICE
    if TL_BACKEND.lower() == "pytorch":
        try:
            tl.set_backend("pytorch")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if DEVICE == "cuda":
                torch.set_num_threads(1)
                print("TensorLy backend: PyTorch (CUDA)")
            else:
                print("TensorLy backend: PyTorch (CPU)")
        except Exception as e:
            print(f"PyTorch backend not available ({e}). Falling back to NumPy.")
            tl.set_backend("numpy")
            DEVICE = "cpu"
            print("TensorLy backend: NumPy")
    else:
        tl.set_backend("numpy")
        DEVICE = "cpu"
        print("TensorLy backend: NumPy")

_set_tl_backend()

def _to_backend(x, use_float64=False):
    """Convert numpy array to the active backend tensor."""
    arr = np.asarray(
        x,
        dtype=np.float64 if (tl.get_backend() == "pytorch" and use_float64) else np.float32,
        order="C",
    )
    if tl.get_backend() == "pytorch":
        return torch.from_numpy(arr).to(DEVICE, non_blocking=True).contiguous()
    else:
        return tl.tensor(arr)

def _to_numpy(x):
    """Convert backend tensor to numpy."""
    if tl.get_backend() == "pytorch":
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return tl.to_numpy(x)

def sanity_report(X, name):
    print(f"[{name}] N={X.shape[0]}, NaNs={np.isnan(X).any()}, Infs={np.isinf(X).any()}")
    if X.size:
        print(f"[{name}] min={np.nanmin(X):.6f}, max={np.nanmax(X):.6f}")

def clean_tiles(X):
    """Replace NaN/Inf and clamp outliers."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -50.0, 50.0)

# Per-band standardization
def fit_band_standardizer(X_train):
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_band_standardizer(X, mu, sigma):
    Z = (X - mu) / sigma
    return np.clip(Z, -20.0, 20.0)

# Random subset helper
def _random_subset_indices(n_total, n_keep=None, seed=REDUCE_SEED):
    if n_keep is None or n_keep > n_total:
        return np.arange(n_total)
    rs = np.random.RandomState(seed)
    return rs.choice(n_total, size=n_keep, replace=False)

# I/O with optional sampling
def readData(directory, max_files=None, random_state=REDUCE_SEED):
    """
    Load typical-only *.npy tiles from directory.
    If max_files is provided, draw a random subset.
    """
    directory = os.fsencode(directory)
    all_files = [f for f in os.listdir(directory) if os.fsdecode(f).endswith(".npy")]
    sel_idx = _random_subset_indices(len(all_files), max_files, seed=random_state)

    filelist = [all_files[i] for i in sel_idx]

    numFiles = len(filelist)
    data_set = np.ones([numFiles, 64, 64, 6], dtype=np.float32)
    true_labels = []
    i = 0
    base = os.fsdecode(directory)
    for f in filelist:
        filename = os.fsdecode(f)
        true_labels.append(1)
        img_array = np.load(os.path.join(base, filename)).astype(np.float32)
        img_array = img_array / 255.0
        data_set[i, :, :, :] = img_array
        i += 1
    return data_set, true_labels

def readData_test(typical_dir, anomaly_dir, max_typical=None, max_anomaly=None, random_state=REDUCE_SEED):
    """
    Load labeled test pool from typical and anomaly dirs.
    If max_* is set, sample that many files from each split.
    """
    typical_dir = os.fsdecode(typical_dir)
    anomaly_dir = os.fsdecode(anomaly_dir)
    typical_files_all = sorted([f for f in os.listdir(typical_dir) if f.endswith(".npy")])
    anomaly_files_all = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(".npy")])

    # Random subset selection
    typ_idx = _random_subset_indices(len(typical_files_all), max_typical, seed=random_state)
    ano_idx = _random_subset_indices(len(anomaly_files_all), max_anomaly, seed=random_state + 1)

    typical_files = [typical_files_all[i] for i in typ_idx]
    anomaly_files = [anomaly_files_all[i] for i in ano_idx]

    n_typ = len(typical_files)
    n_ano = len(anomaly_files)
    n = n_typ + n_ano
    data_set = np.zeros((n, 64, 64, 6), dtype=np.float32)
    true_labels = np.zeros((n,), dtype=np.int32)

    for i, fname in enumerate(typical_files):
        arr = np.load(os.path.join(typical_dir, fname)).astype(np.float32)
        data_set[i] = arr / 255.0
        true_labels[i] = 1

    offset = n_typ
    for j, fname in enumerate(anomaly_files):
        arr = np.load(os.path.join(anomaly_dir, fname)).astype(np.float32)
        data_set[offset + j] = arr / 255.0
        true_labels[offset + j] = -1

    return data_set, true_labels

def _dir_has_npy(d):
    try:
        return os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d))
    except Exception:
        return False

# CP/Tucker decompose (robust)
def decompose_tensor_tucker(tensor, rank, *, mode="hosvd_fast"):
    """
    Fast Tucker per tile.
      mode="hosvd_fast": single-pass HOSVD (n_iter_max=1, init="svd")
      mode="iter":       iterative ALS (n_iter_max=100)
    Returns: (core, factors) as numpy arrays.
    """
    ranks = tuple(rank) if isinstance(rank, (list, tuple, np.ndarray)) else rank

    # Prefer the active backend (float32 on GPU for speed)
    try:
        Xb = _to_backend(tensor, use_float64=False)
        if mode == "hosvd_fast":
            core, factors = _tl_tucker(Xb, ranks, init="svd", n_iter_max=1, tol=0)
        else:
            core, factors = _tl_tucker(Xb, ranks, init="svd", n_iter_max=100, tol=1e-4)
        core_np = _to_numpy(core)
        facs_np = [_to_numpy(Fm) for Fm in factors]
        return core_np.astype(np.float32), [Fm.astype(np.float32) for Fm in facs_np]
    except Exception:
        # Fallback to NumPy backend if the fast path fails
        old = tl.get_backend()
        try:
            tl.set_backend("numpy")
            if mode == "hosvd_fast":
                core, factors = _tl_tucker(tensor.astype(np.float32), ranks, init="svd", n_iter_max=1, tol=0)
            else:
                core, factors = _tl_tucker(tensor.astype(np.float32), ranks, init="svd", n_iter_max=100, tol=1e-4)
            return core.astype(np.float32), [Fm.astype(np.float32) for Fm in factors]
        finally:
            tl.set_backend(old)


def decompose_tensor_parafac(tensor, rank, debug=False):
    try:
        # Make sure we're in the PyTorch backend for GPU work
        if tl.get_backend() != "pytorch":
            tl.set_backend("pytorch")

        Xb = _to_backend(tensor, use_float64=False)
        dev = getattr(Xb, "device", None)
        if not (dev and getattr(dev, "type", "") == "cuda"):
            raise RuntimeError("not-cuda-device")

        shape = Xb.shape

        # ---- Tucker-free, device-safe init ----
        # If rank exceeds any mode, avoid SVD (it pads and can create CPU tensors).
        if any(d < rank for d in shape):
            factors0 = [torch.randn(d, rank, device=dev, dtype=Xb.dtype)
                        for d in shape]
            weights0 = torch.ones(rank, device=dev, dtype=Xb.dtype)
            init = (weights0, factors0)  # <-- proper CP init tuple
        else:
            init = "svd"  # safe when no padding needed
        # ---------------------------------------

        weights, factors = _tl_parafac(
            Xb, rank=rank, init=init,
            n_iter_max=150, tol=1e-6,
            normalize_factors=True, random_state=42
        )
        facs_np = [_to_numpy(Fm) for Fm in factors]
        if all(np.all(np.isfinite(Fm)) for Fm in facs_np):
            return facs_np

    except Exception as e:
        if debug:
            print(f"[CP][GPU] exception: {type(e).__name__}: {e} → fallback")

    # CPU fallback unchanged
    try:
        old = tl.get_backend()
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            tensor.astype(np.float32), rank=rank, init="svd",
            n_iter_max=150, tol=1e-6, normalize_factors=True, random_state=42
        )
        tl.set_backend(old)
        return [Fm.astype(np.float32) for Fm in factors]
    except Exception:
        raise RuntimeError("CP failed on all backends for this tile.")


# Feature extractors
def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

def extract_features_tucker_core(core):
    """Flatten only the Tucker core."""
    return core.ravel()

def extract_features_tucker_factors(factors):
    """Flatten and concatenate only the factor matrices."""
    return np.concatenate([F.ravel() for F in factors], axis=0)

# Tensor build/extract pipelines
def buildTensor(X, rank, num_sets, isTuckerDecomposition=True, ordered=False):
    """
    Decompose each tile.
    - With a CUDA backend, stick to a single worker to avoid context chatter.
    - On CPU, threads are fine (BLAS threads ideally set to 1 via env).
    """
    # Pick worker count (let ThreadPool decide)
    max_workers = None

    def decomp(i):
        if isTuckerDecomposition:
            return decompose_tensor_tucker(X[i], rank, mode="hosvd_fast")
        else:
            factors = decompose_tensor_parafac(X[i], rank)
            return factors

    if ordered:
        decomposed_data = [None] * num_sets
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut = {executor.submit(decomp, i): i for i in range(num_sets)}
            for f in as_completed(fut):
                idx = fut[f]
                decomposed_data[idx] = f.result()
        return decomposed_data
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(decomp, range(num_sets)))


def extractFeatures(decomposed_data, num_sets, isTuckerDecomposition=True, feature_mode="both"):
    """
    Extract per-tile features.
      - For Tucker: feature_mode in {"both","core","factors"}.
      - For CP:     always concatenated factors (unchanged).
    """
    if isTuckerDecomposition:
        mode = (feature_mode or "both").lower()
        if mode not in {"both", "core", "factors"}:
            mode = "both"

        def _feat_tucker(i):
            core, factors = decomposed_data[i]
            if mode == "core":
                return extract_features_tucker_core(core)
            elif mode == "factors":
                return extract_features_tucker_factors(factors)
            else:  # "both"
                return np.concatenate([extract_features_tucker_core(core),
                                       extract_features_tucker_factors(factors)], axis=0)

        with ThreadPoolExecutor() as executor:
            features = list(executor.map(_feat_tucker, range(num_sets)))
        return np.array(features)

    else:
        # CP path unchanged: concat factors
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(
                lambda i: extract_features_cp(decomposed_data[i]),
                range(num_sets)
            ))
        return np.array(features)


# Evaluation helpers
def manual_auc(y_true, scores, positive_label=-1):
    """
    Manual ROC AUC: probability a random positive has a higher score than a random negative.
    Ties count as 0.5. Returns NaN if one class is missing or scores are not finite.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    pos = (y_true == positive_label)
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos_scores = scores[pos]
    neg_scores = scores[~pos]
    greater = (pos_scores[:, None] >  neg_scores[None, :]).sum()
    ties    = (pos_scores[:, None] == neg_scores[None, :]).sum()
    auc = (greater + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)

def _pick_threshold_max_accuracy(y_true, scores, positive_label=-1):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    if thresholds.size == 0:
        thresholds = np.array([np.median(scores)])
    best_th = thresholds[0]; best_acc = -1.0
    for th in thresholds:
        y_pred = np.where(scores >= th, positive_label, -positive_label)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc; best_th = th
    return float(best_th), float(best_acc)


def show_raw_tiles_grayscale(
    X,
    idxs,
    bands=(0,1,2,3,4,5),
    robust_percentiles=(1, 99),
):
    """
    Display selected tiles (rows) and bands (columns) in grayscale only,
    with no titles or labels.
    """
    X = np.asarray(X)
    N = X.shape[0]
    idxs = [min(max(int(i), 0), N - 1) for i in idxs]  # clip to valid range

    n_rows = len(idxs)
    n_cols = len(bands)

    # Shared robust scaling per band
    lo_p, hi_p = robust_percentiles
    band_lims = {}
    for b in bands:
        stack = np.stack([X[i, :, :, b] for i in idxs], axis=0)
        lo = np.percentile(stack, lo_p)
        hi = np.percentile(stack, hi_p)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(stack))
            hi = float(np.nanmax(stack))
            if hi <= lo:
                hi = lo + 1e-6
        band_lims[b] = (lo, hi)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows), squeeze=False
    )
    for r, i_tile in enumerate(idxs):
        for c, b in enumerate(bands):
            ax = axes[r, c]
            vmin, vmax = band_lims[b]
            ax.imshow(
                X[i_tile, :, :, b],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest"
            )
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def cp_reconstruct_tile(A, B, C, h):
    """
    Reconstruct one tile from CP basis (A,B,C) and sample coefficients h.
    Shapes: A:(I,R), B:(J,R), C:(K,R), h:(R,) -> Xhat:(I,J,K)
    """
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C); h = np.asarray(h).reshape(-1)
    # Scale columns of A by h_r, then sum a_r ⊗ b_r ⊗ c_r over r
    Ah = A * h[np.newaxis, :]
    Xhat = np.einsum('ir,jr,kr->ijk', Ah, B, C, optimize=True)
    return Xhat


def visualize_cp_reconstruction(A, B, C, H, idxs, bands=(0,1,2,3,4,5), robust_percentiles=(1, 99)):
    """
    CP reconstruction viewer (grayscale, no labels).
    Rows = selected tile indices; Columns = selected bands.

    Parameters
    ----------
    A, B, C : np.ndarray
        CP factor matrices with shapes (I,R), (J,R), (K,R).
    H : array-like, shape (N,R)
        Coefficients per tile.
    idxs : list[int]
        Exact tile indices to reconstruct and display (e.g., [i, j, k]).
    bands : tuple[int]
        Bands to show as columns.
    robust_percentiles : (float, float)
        (low, high) percentiles for per-band shared scaling across rows.
    """
    H = np.asarray(H)
    N = H.shape[0]
    idxs = [min(max(int(i), 0), N - 1) for i in idxs]  # clip

    # Reconstruct selected tiles
    Xhats = []
    for i in idxs:
        h = H[i]
        Xhat = cp_reconstruct_tile(A, B, C, h)
        Xhats.append(Xhat)

    # Shared robust scaling per band across *reconstructions*
    lo_p, hi_p = robust_percentiles
    band_lims = {}
    for b in bands:
        stack = np.stack([X[:, :, b] for X in Xhats], axis=0)
        lo = np.percentile(stack, lo_p)
        hi = np.percentile(stack, hi_p)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(stack))
            hi = float(np.nanmax(stack))
            if hi <= lo:
                hi = lo + 1e-6
        band_lims[b] = (lo, hi)

    # Plot grid (no titles/labels)
    n_rows, n_cols = len(idxs), len(bands)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows), squeeze=False)
    for r, Xhat in enumerate(Xhats):
        for c, b in enumerate(bands):
            ax = axes[r, c]
            vmin, vmax = band_lims[b]
            ax.imshow(Xhat[:, :, b], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")
    plt.tight_layout()
    plt.show()



def _unpack_tucker_from_decomp(dec):
    """
    Accept (core, [U1,U2,U3]) or (core, (U1,U2,U3)) and return (core, U1, U2, U3) as np.float32.
    U1:(I,r1), U2:(J,r2), U3:(K,r3)
    """
    if not (isinstance(dec, (list, tuple)) and len(dec) == 2):
        raise ValueError("Unexpected Tucker decomposition format; expected (core, factors).")
    core, factors = dec
    if not (isinstance(factors, (list, tuple)) and len(factors) == 3):
        raise ValueError("Unexpected Tucker factors; expected [U1, U2, U3].")
    U1, U2, U3 = factors
    return (np.asarray(core, dtype=np.float32),
            np.asarray(U1,   dtype=np.float32),
            np.asarray(U2,   dtype=np.float32),
            np.asarray(U3,   dtype=np.float32))


def tucker_reconstruct_tile(core, U1, U2, U3):
    """
    Reconstruct a single tile from Tucker (core, U1, U2, U3).
    Shapes: core:(r1,r2,r3), U1:(I,r1), U2:(J,r2), U3:(K,r3) -> Xhat:(I,J,K)
    """
    # Xhat = core ×1 U1 ×2 U2 ×3 U3
    return np.einsum('abc,ia,jb,kc->ijk', core, U1, U2, U3, optimize=True)


def visualize_tucker_reconstruction_per_tile(
    decomp_list,
    idxs,
    bands=(0, 1, 2, 3, 4, 5),
    robust_percentiles=(1, 99),
):
    """
    Tucker reconstruction viewer (grayscale, no labels).
    Rows = selected tile indices; Columns = selected bands.

    Parameters
    ----------
    decomp_list : list of (core, [U1, U2, U3])
        Output from buildTensor(..., isTuckerDecomposition=True).
    idxs : list[int]
        Exact tile indices to reconstruct and display (e.g., [i, j, k]).
    bands : tuple[int]
        Bands to show as columns.
    robust_percentiles : (float, float)
        (low, high) percentiles for per-band shared scaling across rows.
    """
    # Reconstruct selected tiles
    idxs = [int(i) for i in idxs]
    Xhats = []
    for i in idxs:
        core, U1, U2, U3 = _unpack_tucker_from_decomp(decomp_list[i])
        Xhat = tucker_reconstruct_tile(core, U1, U2, U3)
        Xhats.append(Xhat)

    # Shared robust scaling per band across reconstructions
    lo_p, hi_p = robust_percentiles
    band_lims = {}
    for b in bands:
        stack = np.stack([X[:, :, b] for X in Xhats], axis=0)
        lo = np.percentile(stack, lo_p)
        hi = np.percentile(stack, hi_p)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(stack))
            hi = float(np.nanmax(stack))
            if hi <= lo:
                hi = lo + 1e-6
        band_lims[b] = (lo, hi)

    # Plot grid (no titles/labels)
    n_rows, n_cols = len(idxs), len(bands)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows), squeeze=False)
    for r, Xhat in enumerate(Xhats):
        for c, b in enumerate(bands):
            ax = axes[r, c]
            vmin, vmax = band_lims[b]
            ax.imshow(Xhat[:, :, b], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Global CP basis + projection for OC-SVM
def fit_global_cp_basis(X_train, rank, random_state=42, use_gpu=USE_GPU_CP):
    """
    Fit one global CP model to TRAIN tensor (N,64,64,6).
    Returns:
      (A,B,C) as np.float32 and H_train as np.float32 (N,R).
    If subsampling is used, H_train is recomputed for the full TRAIN via projection.
    """
    X_in = X_train

    # GPU branch (TensorLy+PyTorch)
    if use_gpu and tl.get_backend() == "pytorch":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.manual_seed(random_state)
            with torch.no_grad():
                Xb = _to_backend(X_in, use_float64=False)
                weights, factors = _tl_parafac(
                    Xb, rank=rank, init="random",
                    n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
                )
                H_t, A_t, B_t, C_t = factors   # H (N'×R), A (64×R), B (64×R), C (6×R)
                H_t = H_t * weights[None, :]

                # Convert basis to NumPy
                A = A_t.detach().cpu().numpy().astype(np.float32)
                B = B_t.detach().cpu().numpy().astype(np.float32)
                C = C_t.detach().cpu().numpy().astype(np.float32)

                H_np = H_t.detach().cpu().numpy().astype(np.float32)
                return (A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)), H_np

    # CPU fallback (NumPy)
    old_backend = tl.get_backend()
    try:
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            X_in.astype(np.float32), rank=rank, init="random",
            n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
        )
        H, A, B, C = [ np.asarray(F) for F in factors ]
        lam = np.asarray(weights)
        H = H * lam[None, :]
        if X_in.shape[0] != X_train.shape[0]:
            A = A.astype(np.float32); B = B.astype(np.float32); C = C.astype(_np.float32)
            Ginv = precompute_cp_projection(A, B, C)
            H_full = project_cp_coeffs(X_train, A, B, C, Ginv=Ginv)
            return (A, B, C), H_full.astype(np.float32)
        else:
            return (A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)), H.astype(np.float32)
    finally:
        tl.set_backend(old_backend)

def _gram(M):  # (d×R) -> (R×R)
    return M.T @ M

def precompute_cp_projection(A, B, C, eps=1e-8):
    """
    Precompute (M^T M)^{-1}, where M = C ⊙ B ⊙ A (Khatri-Rao).
    M^T M = (A^T A) * (B^T B) * (C^T C) (Hadamard product).
    """
    G = _gram(A) * _gram(B) * _gram(C)
    G = G + eps * np.eye(G.shape[0], dtype=G.dtype)
    Ginv = np.linalg.inv(G)
    return Ginv.astype(np.float64)

def _g_vector_for_tile(X, A, B, C):
    """Compute g_r = <X, a_r ∘ b_r ∘ c_r> for r=1..R via einsum."""
    R = A.shape[1]
    g = np.empty(R, dtype=np.float64)
    for r in range(R):
        g[r] = np.einsum('ijk,i,j,k->', X, A[:, r], B[:, r], C[:, r], optimize=True)
    return g

def project_cp_coeffs(X, A, B, C, Ginv=None):
    """
    Project batch X: (N,64,64,6) onto fixed CP basis (A,B,C).
    Returns H: (N,R) least-squares coefficients.
    """
    if Ginv is None:
        Ginv = precompute_cp_projection(A, B, C)
    N = X.shape[0]; R = A.shape[1]
    H = np.empty((N, R), dtype=np.float64)
    for n in range(N):
        g = _g_vector_for_tile(X[n], A, B, C)  # (R,)
        H[n] = Ginv @ g
    return H.astype(np.float32)


def ocsvm_gamma_grid_for_dim(d):
    base = 1.0 / max(d, 1)
    return [base * t for t in (0.1, 0.3, 1.0, 3.0, 10.0)]

# Read everything once & freeze the split
def prepare_data_once(val_fraction=VAL_FRACTION, random_state=42, split_seed=1):
    """
    Load TRAIN / VAL / TEST (with optional reduction) once and
    return a dict to pass to the pipelines.

    Keys:
      X_train, X_val, X_fin, y_val (or None), y_fin
    """
    # TRAIN (typical-only)
    X_train, _ = readData(
        train_data,
        max_files=(REDUCE_TRAIN_N if REDUCE_DATASETS else None),
        random_state=split_seed
    )

    # TEST pool (typical + anomalies)
    X_pool, y_pool = readData_test(
        test_typical_data, test_anomaly_data,
        max_typical=(REDUCE_TEST_TYP_N if REDUCE_DATASETS else None),
        max_anomaly=(REDUCE_TEST_ANO_N if REDUCE_DATASETS else None),
        random_state=split_seed
    )
    y_pool = np.asarray(y_pool, int)

    # VALIDATION
    if _dir_has_npy(validation_data):
        X_val, _ = readData(
            validation_data,
            max_files=(REDUCE_VAL_N if REDUCE_DATASETS else None),
            random_state=split_seed + 7
        )
        y_val = None
        X_fin, y_fin = X_pool, y_pool
        #print(f"Validation: using separate directory {validation_data} (N={X_val.shape[0]})")
    else:
        print("Validation: no separate dir; splitting the test pool into VAL/FINAL.")
        rng = np.random.RandomState(random_state)
        idx_all = np.arange(len(y_pool)); rng.shuffle(idx_all)
        X_pool = X_pool[idx_all]; y_pool = y_pool[idx_all]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction, random_state=random_state)
        val_idx, final_idx = next(sss.split(np.zeros_like(y_pool), y_pool))
        X_val, y_val = X_pool[val_idx], y_pool[val_idx]
        X_fin, y_fin = X_pool[final_idx], y_pool[final_idx]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_fin": X_fin,
        "y_val": y_val,   # None if separate typical-only VAL dir exists
        "y_fin": y_fin
    }

# Common split fetch + optional band standardization
def get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE):
    """
    Return (X_train, X_val, X_fin, y_val, y_fin, mu_b, sig_b).
    X_* can be standardized using TRAIN stats if requested.
    """
    X_train = np.asarray(data_bundle["X_train"], dtype=np.float32)
    X_val   = np.asarray(data_bundle["X_val"],   dtype=np.float32)
    X_fin   = np.asarray(data_bundle["X_fin"],   dtype=np.float32)
    y_val   = data_bundle.get("y_val", None)
    y_fin   = np.asarray(data_bundle["y_fin"],   dtype=int)

    mu_b = sig_b = None
    if standardize:
        mu_b, sig_b = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu_b, sig_b)
        X_val   = apply_band_standardizer(X_val,   mu_b, sig_b)
        X_fin   = apply_band_standardizer(X_fin,   mu_b, sig_b)

    sanity_report(X_train, "TRAIN")
    sanity_report(X_val,   "VAL")
    sanity_report(X_fin,   "FINAL")

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("TRAIN: NaN/Inf found; cleaning tiles.")
        X_train = clean_tiles(X_train)
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("VAL: NaN/Inf found; cleaning tiles.")
        X_val = clean_tiles(X_val)
    if np.isnan(X_fin).any() or np.isinf(X_fin).any():
        print("FINAL: NaN/Inf found; cleaning tiles.")
        X_fin = clean_tiles(X_fin)

    return X_train, X_val, X_fin, y_val, y_fin, mu_b, sig_b

# Single place for CP fit + project to H
def cp_fit_and_project(X_train, X_val, X_fin, rank, random_state=42):
    """
    Fit a global CP model on TRAIN, then project VAL/FINAL into H.
    Returns: (A,B,C), H_train, H_val, H_fin
    """
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state
    )
    Ginv = precompute_cp_projection(A, B, C)
    H_val = project_cp_coeffs(X_val, A, B, C, Ginv=Ginv)
    H_fin = project_cp_coeffs(X_fin, A, B, C, Ginv=Ginv)
    return (A, B, C), H_train, H_val, H_fin


# OC-SVM using preloaded data
def OC_SVM(type, Htr_w, Hval_w, Hfin_w):
    start_time = time.process_time()

    feat_dim = Htr_w.shape[1]

    # OC-SVM search (gamma scaled by 1/d)
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(feat_dim) + ["scale", "auto"],
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    best_score_tuple = None
    best_model = None
    best_params = None
    best_aux_print = ""

    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params).fit(Htr_w)
        s_val = -model.decision_function(Hval_w).ravel()  # larger => more anomalous
        if not np.all(np.isfinite(s_val)):
            continue

        # Typical-only VAL: minimize FP@0, tie-break on P95 and mean
        fp_rate = float((s_val >= 0.0).mean())
        p95 = float(np.percentile(s_val, 95))
        mean_s = float(np.mean(s_val))
        score_tuple = (fp_rate, p95, mean_s)
        aux = f"FP={fp_rate:.3f}, P95={p95:.4f}, mean={mean_s:.4f}"

        if (best_score_tuple is None) or (score_tuple < best_score_tuple):
            best_score_tuple = score_tuple
            best_model = model
            best_params = params
            best_aux_print = aux

    print(f"[{type}+OCSVM] (VAL one-class) chose {best_params} best_obj:({best_aux_print})"
          f" Elapsed: {round(time.process_time() - start_time, 2)}")

    return best_score_tuple, best_model, best_params, best_aux_print


def ocsvm_only(
    Ztr, Zv, Zte,
    displayConfusionMatrix=False):
    """
    OC-SVM on raw (flattened) tiles, no CP.
    - Expects `data_bundle` from your prepare_data_once(...) path.
    - VAL is typical-only → choose params by minimizing FP@0 (ties: P95, mean).
    - No robust fallbacks.
    - Optional PCA (dim reduction). Set do_pca=False to disable PCA entirely.
    Returns: (accuracy, auc) on FINAL.
    """
    start_time = time.time()

    # --- Hyperparameter grid (VAL is typical-only) ---
    d = Ztr.shape[1]
    param_grid = {
        "kernel": ["rbf"],
        "gamma": ocsvm_gamma_grid_for_dim(d) + ["scale", "auto"],
        "nu": [0.01, 0.02, 0.05, 0.10, 0.20],
    }

    best_model = None
    best_params = None
    best_aux_print = None
    best_score_tuple = None

    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params).fit(Ztr)
        s_val = -model.decision_function(Zv).ravel()  # larger => more anomalous

        # Typical-only VAL: minimize FP@0
        fp_rate = float((s_val >= 0.0).mean())
        p95 = float(np.percentile(s_val, 95))
        mean_s = float(np.mean(s_val))
        score_tuple = (fp_rate, p95, mean_s)
        aux = f"FP={fp_rate:.3f}, P95={p95:.4f}, mean={mean_s:.4f}"

        if (best_score_tuple is None) or (score_tuple < best_score_tuple):
            best_score_tuple = score_tuple
            best_model = model
            best_params = params
            best_aux_print = aux

    if best_model is None:
        raise RuntimeError("OC-SVM grid search failed to select a model.")

    print(f"OCSVM (VAL one-class) chose {best_params} ({best_aux_print})")

    # --- FINAL evaluation ---
    s_fin  = -best_model.decision_function(Zte).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"OCSVM FINAL AUC={auc_fin:.3f}")
    print(f"OCSVM threshold={th_opt:.6f} | accuracy={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(s_fin >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"]).plot()
        plt.show()

    print("Elapsed:", round(time.time() - start_time, 2), "s")
    return float(acc_opt), float(auc_fin)


def one_class_svm(Ztr, Zv, Zte):
    print("One-class SVM (raw pixels)")
    acc, auc = ocsvm_only(Ztr, Zv, Zte)
    print("One-class SVM best accuracy:", acc, "auc:", auc)


def autoencoder_param_search(type, Htr_w, Hval_w, Hfin_w):
    best_err_va = None
    best_autoencoder = None
    best_param = None
    best_run_time = None
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            sum_err_va, autoencoder, run_time = neural_network_autoencoder(factor, bottleneck, Htr_w, Hval_w, Hfin_w)
            if (best_err_va is None or sum_err_va < best_err_va):
                best_err_va = sum_err_va
                best_autoencoder = autoencoder
                best_param = (factor, bottleneck)
                best_run_time = run_time

    print(f'[{type}+AE] best_param={best_param} best_obj={best_err_va}'
          f" Elapsed: {best_run_time}" )

    return best_err_va, best_autoencoder, best_run_time


def autoencoder_anomaly(Z_tr, Z_va, Z_fi, factor, bottleneck):
    # AE definition
    input_dim = Z_tr.shape[1]
    inp = Input(shape=(input_dim,))
    enc = Dense(128 * factor, activation='relu')(inp)
    enc = Dropout(0.1)(enc)
    bott = Dense(bottleneck, activation='relu')(enc)
    dec = Dense(128 * factor, activation='relu')(bott)
    dec = Dropout(0.1)(dec)
    out = Dense(input_dim, activation='sigmoid')(dec)
    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    autoencoder.fit(Z_tr, Z_tr, epochs=10, batch_size=32,
                    validation_data=(Z_va, Z_va),
                    callbacks=[es], verbose=0)


    # Threshold from VAL (typical-only)
    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean(np.square(Z_va - recon_va), axis=1)

    return np.sum(err_va), autoencoder, Z_fi, y_fin


def autoencoder(Ztr, Zv, Zte, sweep_factors=(1, 2, 3), sweep_bottlenecks=(16, 32, 64)):
    """
    Hyperparam sweep for the raw-pixel autoencoder (no CP/Tucker).
    """
    print("Autoencoder (raw-pixel) — sweeping factors and bottlenecks")
    rank_autoencoder = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_err_va = None
    best_rank = None

    for factor in sweep_factors:
        for bottleneck in sweep_bottlenecks:
            err_va, autoencoder, Z_fi, y_fin = autoencoder_anomaly(Ztr, Zv, Zte, factor, bottleneck)

            if (rank_err_va is None or err_va < rank_err_va):
                rank_err_va = err_va
                rank_autoencoder = autoencoder
                rank_y_fin = y_fin
                rank_Z_fi = Z_fi
                best_rank = (factor, bottleneck)

                # Intermediate result
                recon_te = rank_autoencoder.predict(rank_Z_fi, verbose=0)
                err_te = np.mean(np.square(rank_Z_fi - recon_te), axis=1)  # anomaly-positive scores
                auc_fin = manual_auc(rank_y_fin, err_te, positive_label=-1)
                th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_te, positive_label=-1)
                print("Factor:", factor, "Bottleneck:", bottleneck)
                print(f"AE Intermediate result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")

    # Intermediate result
    recon_te = rank_autoencoder.predict(rank_Z_fi, verbose=0)
    err_te = np.mean(np.square(rank_Z_fi - recon_te), axis=1)  # anomaly-positive scores
    auc_fin = manual_auc(rank_y_fin, err_te, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_te, positive_label=-1)
    print(f"AE Final result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")


def cp_rank_search_autoencoder(data_bundle):
    print("CP rank search (autoencoder)")
    rank_autoencoder = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_err_va = None
    best_rank = None

    startRank = 10
    endRank = 385
    step = 5
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            for i in range(startRank, endRank, step):
                rank = i
                print("Factor:", factor, "Bottleneck:", bottleneck, "Rank:", i)
                with peak_ram(prefix=f"CP+AE", label=f"R={rank}", interval=0.02) as m:
                    err_va, autoencoder, Z_fi, y_fin = parafac_autoencoder(rank, factor, bottleneck, data_bundle)

                if (rank_err_va is None or err_va < rank_err_va):
                    rank_err_va = err_va
                    rank_autoencoder = autoencoder
                    rank_y_fin = y_fin
                    rank_Z_fi = Z_fi
                    best_rank= (rank, factor, bottleneck)

                    # Intermediate results
                    recon_fi = rank_autoencoder.predict(rank_Z_fi, verbose=0)
                    err_fi = np.mean(np.square(rank_Z_fi - recon_fi), axis=1)  # anomaly-positive scores
                    auc_fin = manual_auc(rank_y_fin, err_fi, positive_label=-1)
                    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_fi, positive_label=-1)
                    print(f"[CP+AE] (global) Intermediate result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")

    # Score FINAL
    recon_fi = rank_autoencoder.predict(rank_Z_fi, verbose=0)
    err_fi = np.mean(np.square(rank_Z_fi - recon_fi), axis=1)  # anomaly-positive scores
    auc_fin = manual_auc(rank_y_fin, err_fi, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_fi, positive_label=-1)
    print(f"[CP+AE] (global) Final result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")


#
# Tucker with Autoencoder (shared path)
#
def neural_network_autoencoder(factor, bottleneck, Z_tr, Z_va, Z_fi):
    start_time = time.process_time()

    # Define the autoencoder model
    input_dim = Z_tr.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)

    # Decoder
    decoder = Dense(128 * factor, activation='relu')(encoder)
    # keep variable names; switch head to linear unless your inputs are min-maxed to [0,1]
    decoder = Dense(input_dim, activation='linear')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback (use VAL split)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(
        Z_tr, Z_tr,
        epochs=100, batch_size=32,
        validation_data=(Z_va, Z_va),
        callbacks=[early_stopping],
        verbose=0
    )

    # --- anomaly-positive scores: per-row MSE on TRAIN and VAL ---
    recon_tr = autoencoder.predict(Z_tr, verbose=0)
    s_tr = np.mean((Z_tr - recon_tr) ** 2, axis=1).astype(float)

    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean((Z_va - recon_va) ** 2, axis=1).astype(float)  # keep name; this is s_va

    if not (np.all(np.isfinite(s_tr)) and np.all(np.isfinite(err_va))):
        raise RuntimeError("Non-finite AE scores; check inputs/scaling.")

    # --- TRAIN-anchored multi-quantile FP objective (lower is better) ---
    q_grid = (90, 95, 97.5, 99, 99.5, 99.9)
    fp_vals = []
    for q in q_grid:
        t = float(np.percentile(s_tr, q))
        fp = float(np.mean(err_va >= t))   # VAL is typical-only → this is FPR at that tail
        fp_vals.append(fp)
    best_obj_value = float(np.mean(fp_vals))

    # Threshold from VAL (typical-only directory) for a target FP
    #VAL_FP_TARGET = 0.05 if 'VAL_FP_TARGET' not in globals() else VAL_FP_TARGET
    threshold = float(np.percentile(err_va, 100.0 * (1.0 - float(VAL_FP_TARGET))))

    runTime = round(time.process_time() - start_time, 2)

    return best_obj_value, autoencoder, runTime



def tucker_rank_search_autoencoder(data_bundle):
    """
    Sweep ranks; report AUC by (rank, factor, bottleneck).
    """
    print("Tucker rank search (autoencoder)")

    rank_autoencoder = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_err_va = None
    best_rank = None

    rankSet = sorted({5, 16, 32, 64})
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            for i in rankSet:
                for j in rankSet:
                    for k in sorted({5, 16}):
                        rank = (i, j, k)
                        print("Rank:", i, j, k, "Factor", factor, "Bottleneck:", bottleneck)
                        with peak_ram(prefix=f"Tucker+AE", label=f"R={rank}", interval=0.02) as m:
                            err_va, autoencoder, Z_fi, y_fin = tucker_neural_network_autoencoder(rank, factor, bottleneck, data_bundle,
                                                                          feature_mode=TUCKER_FEATURE_MODE)

                        if (rank_err_va is None or err_va < rank_err_va):
                            rank_err_va = err_va
                            rank_autoencoder = autoencoder
                            rank_y_fin = y_fin
                            rank_Z_fi = Z_fi
                            best_rank = (rank, factor, bottleneck)

                            # Intermediate prediction
                            recon_fi = rank_autoencoder.predict(rank_Z_fi, verbose=0)
                            err_fi = np.mean(np.square(rank_Z_fi - recon_fi), axis=1)  # anomaly-positive scores
                            # AUC + max-accuracy threshold (for parity with other strategies)
                            auc_fin = manual_auc(rank_y_fin, err_fi, positive_label=-1)
                            th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_fi, positive_label=-1)
                            print(f"[Tucker+AE] Intermediate result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")


    # Predict on FINAL
    recon_fi = rank_autoencoder.predict(rank_Z_fi, verbose=0)
    err_fi = np.mean(np.square(rank_Z_fi - recon_fi), axis=1)  # anomaly-positive scores
    # AUC + max-accuracy threshold (for parity with other strategies)
    auc_fin = manual_auc(rank_y_fin, err_fi, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_fi, positive_label=-1)
    print(f"Tucker+AE Final result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")

    return acc_opt, auc_fin


def _if_mean_score_scorer(estimator, X, y=None):
    # Higher is better (less "anomalous" on average)
    try:
        return float(np.mean(estimator.score_samples(X)))
    except Exception:
        return -np.inf

def isolation_forest(model_tag, Htr_w, Hval_w, Hfin_w, random_state=42):
    """
    Tune IF by minimizing the average VAL false-positive rate across a range of
    TRAIN-anchored thresholds (quantiles of -score_samples on TRAIN).
    Then calibrate the final threshold on VAL at (1 - VAL_FP_TARGET).
    """
    start_time = time.process_time()
    warnings.filterwarnings('ignore', category=UserWarning)
    assert (Hval_w is not None) and (Hval_w.shape[0] > 0)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples':  [0.5, 0.75, 1.0],
        'max_features': [0.5, 0.75, 1.0],
        'bootstrap':    [False, True],
        'contamination':[0.05, 0.10, 0.20],  # inert (we use score_samples), harmless to keep
        'random_state': [random_state],
        'n_jobs':       [-1],
    }

    # Quantiles to sweep (TRAIN-anchored). Adjust granularity if you like.
    q_grid = [90, 95, 97.5, 99, 99.5, 99.9]

    best_if, best_params, best_obj, thr = None, None, np.inf, None
    any_finite = False

    for p in ParameterGrid(param_grid):
        clf = IsolationForest(**p).fit(Htr_w)

        s_tr = -clf.score_samples(Htr_w)
        s_va = -clf.score_samples(Hval_w)
        if not (np.all(np.isfinite(s_tr)) and np.all(np.isfinite(s_va))):
            continue
        any_finite = True

        # Average VAL FP across a range of TRAIN-anchored thresholds
        fp_vals = []
        for q in q_grid:
            t = float(np.percentile(s_tr, q))
            fp = float(np.mean(s_va >= t))
            fp_vals.append(fp)

        # Small regularizer to avoid needlessly huge models; tweak or drop as desired
        reg = 1e-4 * p['n_estimators']
        obj = float(np.mean(fp_vals)) + reg

        if obj < best_obj:
            best_obj = obj
            best_if = clf
            best_params = dict(p)
            # Final threshold: hit VAL_FP_TARGET on VAL (typical-only)
            q_star = 100.0 * (1.0 - float(VAL_FP_TARGET))
            thr = float(np.percentile(s_va, q_star))

    if not any_finite or best_if is None or thr is None:
        raise RuntimeError("IF grid produced no finite scores; check data for NaN/inf.")

    elapsed = round(time.process_time() - start_time, 2)
    print(f"[{model_tag}+IF] best_obj={best_obj:.6f} best_param: {best_params} Elapsed (CPU s): {elapsed}")
    return best_if, best_obj, thr, best_params


def cp_rank_search_isolation_forest(data_bundle):
    print("CP rank search (Isolation Forest)")
    rank_best_if = None
    rank_best_obj = None
    rank_best_thr = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_best_param = None
    rank_best_rank = None

    startRank = 10; endRank = 385; step = 5
    for rank in range(startRank, endRank, step):
        print("Rank:", rank)
        with peak_ram(prefix=f"CP+IF", label=f"R={rank}", interval=0.02) as m:
            best_if, best_obj, thr, Z_fi, y_fin, best_params = parafac_isolation_forest(rank, data_bundle, displayConfusionMatrix=False)
        if rank_best_obj is None or best_obj < rank_best_obj:
            rank_best_obj = best_obj
            rank_best_if = best_if
            rank_best_thr = thr
            rank_Z_fi = Z_fi
            rank_y_fin = y_fin
            rank_best_param = best_params
            rank_best_rank = rank

            # --- Intermediate scoring ---
            s_fi = -rank_best_if.score_samples(rank_Z_fi)  # anomaly-positive
            preds = np.where(s_fi >= rank_best_thr, -1, 1)
            acc = float(np.mean(preds == rank_y_fin))
            auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
            print(f"[CP+IF] Intermediate result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
                  f"| params={rank_best_param} "
                  f"| thr={rank_best_thr:.6f} "
                  f"| target_FP={VAL_FP_TARGET:.3f}")

    # --- FINAL scoring ---
    s_fi = -rank_best_if.score_samples(rank_Z_fi)  # anomaly-positive
    preds = np.where(s_fi >= rank_best_thr, -1, 1)
    acc = float(np.mean(preds == rank_y_fin))
    auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
    print(f"[CP+IF] Final result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
          f"| params={rank_best_param} "
          f"| thr={rank_best_thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")


# Raw-pixel IsolationForest (no decomposition), using shared data path
def isolation_forest_anomaly(Z_tr, Z_va, Z_fi, random_state=42):
    warnings.filterwarnings('ignore', category=UserWarning)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples':  [0.5, 0.75, 1.0],
        'contamination':[0.05, 0.10, 0.20],
        'max_features': [0.5, 0.75, 1.0],
        'bootstrap':    [False, True],
        'random_state': [random_state],
        'n_jobs':       [-1],
    }

    best_if = None
    best_params = None
    thr = None

    if USE_VAL_FOR_IF and (y_val is None) and (Z_va is not None) and (Z_va.shape[0] > 0):
        # ------- Typical-only VAL: choose hyperparams by minimizing P95 on VAL; calibrate threshold from VAL -------
        best_obj = np.inf
        for p in ParameterGrid(param_grid):
            clf = IsolationForest(**p).fit(Z_tr)
            s_va = -clf.score_samples(Z_va)  # anomaly-positive
            if not np.all(np.isfinite(s_va)):
                continue
            p95 = float(np.percentile(s_va, 95))
            if p95 < best_obj:
                best_obj = p95
                best_if = clf
                best_params = dict(p)
                # set threshold to achieve target FP on VAL
                q = 100.0 * (1.0 - float(VAL_FP_TARGET))
                thr = float(np.percentile(s_va, q))

    # --- FINAL scoring ---
    s_fi = -best_if.score_samples(Z_fi)  # anomaly-positive
    preds = np.where(s_fi >= thr, -1, 1)
    acc = float(np.mean(preds == y_fin))
    auc_fin = manual_auc(y_fin, s_fi, positive_label=-1)
    print(f"IF Final result AUC={auc_fin} ACC={acc} obj={best_obj}"
          f"| params={best_params} "
          f"| thr={thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")


#
# Tucker + Isolation Forest
#
def tucker_isolation_forests(Z_tr, Z_va, Z_fi, random_state=42):
    start_time = time.process_time()

    # --- grid ---
    warnings.filterwarnings('ignore', category=UserWarning)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples':  [0.5, 0.75, 1.0],
        'contamination':[0.05, 0.10, 0.20],
        'max_features': [0.5, 0.75, 1.0],
        'bootstrap':    [False, True],
        'random_state': [random_state],
        'n_jobs':       [-1],
    }

    best_if = None
    best_params = None
    thr = None

    if USE_VAL_FOR_IF and (y_val is None) and (Z_va is not None) and (Z_va.shape[0] > 0):
        best_obj = np.inf
        for p in ParameterGrid(param_grid):
            clf = IsolationForest(**p).fit(Z_tr)
            s_va = -clf.score_samples(Z_va)  # anomaly-positive
            if not np.all(np.isfinite(s_va)):
                continue
            p95 = float(np.percentile(s_va, 95))
            if p95 < best_obj:
                best_obj = p95
                best_if = clf
                best_params = dict(p)
                q = 100.0 * (1.0 - float(VAL_FP_TARGET))
                thr = float(np.percentile(s_va, q))

    print(f"[Tucker+IF] best_obj:{best_obj} best_param: {best_params} Elapsed: {round(time.process_time() - start_time, 2)}")
    return best_if, best_obj, thr, best_params


def tucker_rank_search_isolation_forest(data_bundle):
    print("Tucker rank search (Isolation Forest)")

    rank_best_if = None
    rank_best_obj = None
    rank_best_thr = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_best_param = None
    rank_best_rank = None

    rankSet = sorted({5, 16, 32, 64})
    for i in rankSet:
        for j in rankSet:
            for k in sorted({5, 16}):  # keep k modest (band mode)
                r = (i, j, k)
                print("Rank:", r)
                best_if, best_obj, thr, Z_fi, y_fin, best_params = tucker_isolation_forests(r, data_bundle, feature_mode=TUCKER_FEATURE_MODE)
                if rank_best_obj is None or best_obj < rank_best_obj:
                    rank_best_obj = best_obj
                    rank_best_if = best_if
                    rank_best_thr = thr
                    rank_Z_fi = Z_fi
                    rank_y_fin = y_fin
                    rank_best_param = best_params
                    rank_best_rank = r

                    # --- Intermediate scoring ---
                    s_fi = -rank_best_if.score_samples(rank_Z_fi)
                    preds = np.where(s_fi >= rank_best_thr, -1, 1)
                    acc = float(np.mean(preds == rank_y_fin))
                    auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
                    print(f"[Tucker+IF] Intermediate result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
                          f"| params={rank_best_param} "
                          f"| thr={rank_best_thr:.6f} "
                          f"| target_FP={VAL_FP_TARGET:.3f}")

    # --- FINAL scoring ---
    s_fi = -rank_best_if.score_samples(rank_Z_fi)
    preds = np.where(s_fi >= rank_best_thr, -1, 1)
    acc = float(np.mean(preds == rank_y_fin))
    auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
    print(f"[Tucker+IF] Final result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
          f"| params={rank_best_param} "
          f"| thr={rank_best_thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")


def _stratified_boot_idxs(y, rng):
    y = np.asarray(y)
    pos = np.where(y == -1)[0]
    neg = np.where(y != -1)[0]
    bpos = rng.choice(pos, size=pos.size, replace=True)
    bneg = rng.choice(neg, size=neg.size, replace=True)
    return np.concatenate([bpos, bneg])

def _bootstrap_metric(y, scores, *, n_boot=1000, ci=95, positive_label=-1,
                      metric="auc", fixed_threshold=None, rng_seed=123):
    """Stratified bootstrap over (y, scores). metric in {'auc','acc'}.
       If metric='acc' and fixed_threshold is None, we re-tune the threshold
       on each bootstrap resample (optimistic but consistent with your current code).
       If fixed_threshold is provided, we evaluate accuracy at that fixed threshold.
    """
    y = np.asarray(y); s = np.asarray(scores, float)
    rng = np.random.RandomState(rng_seed)
    vals = []
    for _ in range(n_boot):
        idx = _stratified_boot_idxs(y, rng)
        yb, sb = y[idx], s[idx]
        if metric == "auc":
            v = manual_auc(yb, sb, positive_label=positive_label)
        elif metric == "acc":
            if fixed_threshold is None:
                th_b, _ = _pick_threshold_max_accuracy(yb, sb, positive_label=positive_label)
                yp = np.where(sb >= th_b, positive_label, -positive_label)
            else:
                yp = np.where(sb >= fixed_threshold, positive_label, -positive_label)
            v = float((yp == yb).mean())
        else:
            raise ValueError("metric must be 'auc' or 'acc'")
        if np.isfinite(v):
            vals.append(v)

    vals = np.asarray(vals, float)
    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan, "low": np.nan, "high": np.nan}

    low_q = (100 - ci) / 2.0
    high_q = 100 - low_q
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
        "low": float(np.percentile(vals, low_q)),
        "high": float(np.percentile(vals, high_q)),
    }


def _flatten(X):
    """Flatten samples to 2D (n_samples, n_features)."""
    X = np.asarray(X)
    return X.reshape((X.shape[0], -1))

def _fit_eval_ocsvm(Htr, Hval, y_val, *, nu, gamma):
    """Fit OC-SVM on Htr; return validation AUC using your conventions."""
    mdl = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    mdl.fit(Htr)
    # Score: make higher score = "more anomalous" to match manual_auc(positive_label=-1)
    s_val = -mdl.decision_function(Hval).ravel()
    auc = manual_auc(y_val, s_val, positive_label=-1)
    return auc, mdl

###
## Evaluate models
###
def evaluate_OC_SVM(type, rank, Hfin_w, y_fin, best_score_tuple, best_model, best_params, best_aux_print):
    # FINAL evaluation
    s_fin = -best_model.decision_function(Hfin_w).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"[{type}+OCSVM] Final result RANK={rank} AUC={auc_fin} | ACC={acc_opt} "
          f"| param=({best_params}) | aux={best_aux_print}")

    # Bootstrap AUC (threshold-free, robust)
    auc_boot = _bootstrap_metric(y_fin, s_fin, n_boot=2000, ci=95, positive_label=-1, metric="auc",
                                 rng_seed=42)
    print(f"[{type}+OCSVM] AUC={auc_fin:.4f} Boot: mean:{auc_boot['mean']:.4f} std:{auc_boot['std']:.4f}, "
          f"CI({auc_boot['low']:.4f}–{auc_boot['high']:.4f}) | ")

def evaluate_AE(type, rank, Hfin_w, y_fin, autoencoder):
    # Score FINAL
    recon_fi = autoencoder.predict(Hfin_w, verbose=0)
    err_fi = np.mean(np.square(Hfin_w - recon_fi), axis=1)  # anomaly-positive scores
    auc_fin = manual_auc(y_fin, err_fi, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_fi, positive_label=-1)
    print(f"[{type}+AE] (global) Final result Rank={rank} AUC={auc_fin} ACC={acc_opt}")

    # Bootstrap AUC (threshold-free, robust)
    auc_boot = _bootstrap_metric(y_fin, err_fi, n_boot=2000, ci=95, positive_label=-1, metric="auc",
                                 rng_seed=42)
    print(f"[{type}+AE] AUC={auc_fin:.4f} Boot: mean:{auc_boot['mean']:.4f} std:{auc_boot['std']:.4f}, "
          f"CI({auc_boot['low']:.4f}–{auc_boot['high']:.4f}) | ")


def evaluate_IF(type, rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params):
    # --- FINAL scoring ---
    s_fi = -best_if.score_samples(Hfin_w)  # anomaly-positive
    preds = np.where(s_fi >= thr, -1, 1)
    acc = float(np.mean(preds == y_fin))
    auc_fin = manual_auc(y_fin, s_fi, positive_label=-1)
    print(f"[{type}+IF] Final result rank={rank} AUC={auc_fin} ACC={acc} obj={best_obj}"
          f"| params={best_params} "
          f"| thr={thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")

    # Bootstrap AUC (threshold-free, robust)
    auc_boot = _bootstrap_metric(y_fin, s_fi, n_boot=2000, ci=95, positive_label=-1, metric="auc",
                                 rng_seed=42)
    print(f"[{type}+IF] AUC={auc_fin:.4f} Boot: mean:{auc_boot['mean']:.4f} std:{auc_boot['std']:.4f}, "
          f"CI({auc_boot['low']:.4f}–{auc_boot['high']:.4f}) | ")


use_pca_whiten = True
# Rank search
if not no_decomposition and not use_predefined_rank:
    print('Rank search')
    for split_seed in {3}:
        print('Split seed:', split_seed)

        # Entry (reads once, then passes data to pipelines)
        data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42, split_seed=split_seed)
        X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

        if enable_pca_oc_svm or enable_pca_autoencoder or enable_pca_isolation_forest:
            #evaluate_pca_oc_svm(X_train, X_val, X_fin, y_val, y_fin)
            # ---- knobs (tweak as needed) ----
            RANDOM_SEED = 42

            # -------- PCA + OC-SVM pipeline --------
            # Flatten
            Xtr = _flatten(X_train)
            Xva = _flatten(X_val)
            Xte = _flatten(X_fin)

            # Standardize BEFORE PCA (important for RBF kernels downstream)
            scaler0 = StandardScaler()
            Xtr_s = scaler0.fit_transform(Xtr)
            Xva_s = scaler0.transform(Xva)
            Xte_s = scaler0.transform(Xte)

            startRank = 10;
            endRank = 385;
            step = 5  # tighter range for speed
            for rank in range(startRank, endRank, step):
                print("Rank:", rank)
                with peak_ram(prefix="PCA only", label=f"rank={rank}", interval=0.02) as m:
                    # Fit PCA on TRAIN only; transform VAL/FINAL
                    pca = PCA(n_components=rank, whiten=use_pca_whiten, svd_solver='auto', random_state=RANDOM_SEED)
                    H_train = pca.fit_transform(Xtr_s)
                    H_val = pca.transform(Xva_s)
                    H_fin = pca.transform(Xte_s)

                    # Scale
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    Htr_s = scaler.fit_transform(H_train)
                    Hval_s = scaler.transform(H_val)
                    Hfin_s = scaler.transform(H_fin)

                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                    if enable_pca_oc_svm:
                        with peak_ram(prefix="PCA+OCSVM", label=f"rank={rank}", interval=0.02) as m:
                            best_score_tuple, best_model, best_params, best_aux_print = OC_SVM('PCA', Htr_w, Hval_w, Hfin_w)
                            evaluate_OC_SVM('PCA', rank, Hfin_w, y_fin, best_score_tuple, best_model, best_params, best_aux_print)

                    if enable_pca_autoencoder:
                        with peak_ram(prefix="PCA+AE", label=f"rank={rank}", interval=0.02) as m:
                            best_err_va, best_autoencoder, best_run_time = autoencoder_param_search('PCA', Htr_w, Hval_w, Hfin_w)
                            evaluate_AE('PCA', rank, Hfin_w, y_fin, best_autoencoder)

                    if enable_pca_isolation_forest:
                        with peak_ram(prefix="PCA+IF", label=f"rank={rank}", interval=0.02) as m:
                            best_if, best_obj, thr, best_params = isolation_forest('PCA', Htr_w, Hval_w, Hfin_w)
                            evaluate_IF('PCA', rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params)


        if enable_cp_oc_svm or enable_cp_autoencoder or enable_cp_isolation_forest:
            startRank = 10
            endRank = 385
            step = 5  # tighter range for speed
            for rank in range(startRank, endRank, step):
                print("Rank:", rank)

                Htr_w, Hval_w, Hfin_w = None, None, None
                with peak_ram(prefix=f"CP only", label=f"R={rank}", interval=0.02) as m:
                    # Global CP fit + project
                    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
                        X_train, X_val, X_fin, rank,
                        random_state=42
                    )

                    # Scale
                    scaler = StandardScaler()
                    Htr_s = scaler.fit_transform(H_train)
                    Hval_s = scaler.transform(H_val)
                    Hfin_s = scaler.transform(H_fin)

                    # Feature pathway: full H (optionally PCA-whiten)
                    if use_pca_whiten:
                        pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                        Htr_w = pca.fit_transform(Htr_s)
                        Hval_w = pca.transform(Hval_s)
                        Hfin_w = pca.transform(Hfin_s)
                    else:
                        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                if enable_cp_oc_svm:
                    with peak_ram(prefix=f"CP+OCSVM", label=f"R={rank}", interval=0.02) as m:
                        best_score_tuple, best_model, best_params, best_aux_print = OC_SVM('CP', Htr_w, Hval_w, Hfin_w)
                        if TRAINING_EVALUATION:
                            evaluate_OC_SVM('CP', rank, Hfin_w, y_fin, best_score_tuple, best_model, best_params, best_aux_print)

                if enable_cp_autoencoder:
                    with peak_ram(prefix=f"CP+AE", label=f"R={rank}", interval=0.02) as m:
                        best_err_va, best_autoencoder, best_run_time = autoencoder_param_search('CP', Htr_w=Htr_w, Hval_w=Hval_w, Hfin_w=Hfin_w)
                        if TRAINING_EVALUATION:
                            evaluate_AE('CP', rank, Hfin_w, y_fin, best_autoencoder)

                if enable_cp_isolation_forest:
                    with peak_ram(prefix=f"CP+IF", label=f"R={rank}", interval=0.02) as m:
                        best_if, best_obj, thr, best_params = isolation_forest('CP', Htr_w=Htr_w, Hval_w=Hval_w, Hfin_w=Hfin_w)
                        if TRAINING_EVALUATION:
                            evaluate_IF('CP', rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params)

        if enable_tucker_oc_svm or enable_tucker_autoencoder or enable_tucker_isolation_forest:
            rankSet = sorted({5, 16, 32, 64})
            for i in rankSet:
                for j in rankSet:
                    for k in sorted({5, 16}):
                        rank = (i, j, k)
                        print("Rank:", i, j, k)

                        Z_tr, Z_va, Z_fi = None, None, None
                        with peak_ram(prefix=f"Tucker only", label=f"R={rank}", interval=0.02) as m:
                            # Tucker decompositions
                            n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
                            decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
                            decomp_va = buildTensor(X_val, rank, n_va, isTuckerDecomposition=True)
                            decomp_fi = buildTensor(X_fin, rank, n_fi, isTuckerDecomposition=True)

                            # Feature extraction + scaling (fit on TRAIN only)
                            Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True,
                                                      feature_mode=TUCKER_FEATURE_MODE)
                            Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True,
                                                      feature_mode=TUCKER_FEATURE_MODE)
                            Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True,
                                                      feature_mode=TUCKER_FEATURE_MODE)

                            # Scale
                            scaler = StandardScaler()
                            Htr_s = scaler.fit_transform(Feat_tr)
                            Hval_s = scaler.transform(Feat_va)
                            Hfin_s = scaler.transform(Feat_fi)

                            # Feature pathway: full H (optionally PCA-whiten)
                            if use_pca_whiten:
                                pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                                Htr_w = pca.fit_transform(Htr_s)
                                Hval_w = pca.transform(Hval_s)
                                Hfin_w = pca.transform(Hfin_s)
                            else:
                                Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                        if enable_tucker_oc_svm:
                            with peak_ram(prefix=f"Tucker+OCSVM", label=f"R={rank}", interval=0.02) as m:
                                best_score_tuple, best_model, best_params, best_aux_print = OC_SVM('Tucker', Htr_w, Hval_w, Hfin_w)
                                if TRAINING_EVALUATION:
                                    evaluate_OC_SVM('Tucker', rank, Hfin_w, y_fin, best_score_tuple, best_model, best_params, best_aux_print)

                        if enable_tucker_autoencoder:
                            with peak_ram(prefix=f"Tucker+AE", label=f"R={rank}", interval=0.02) as m:
                                best_err_va, best_autoencoder, best_run_time = autoencoder_param_search('Tucker', Htr_w, Hval_w, Hfin_w)
                                if TRAINING_EVALUATION:
                                    evaluate_AE('Tucker', rank, Hfin_w, y_fin, best_autoencoder)

                        if enable_tucker_isolation_forest:
                            with peak_ram(prefix=f"Tucker+IF", label=f"R={rank}", interval=0.02) as m:
                                best_if, best_obj, thr, best_params = isolation_forest('Tucker', Htr_w, Hval_w, Hfin_w)
                                if TRAINING_EVALUATION:
                                    evaluate_IF('Tucker', rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params)

if no_decomposition:
    print('No Decomposition')
    for split_seed in {1, 3, 5}:
        print('Split seed:', split_seed)
        # Entry (reads once, then passes data to pipelines)
        data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42, split_seed=split_seed)
        X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

        # --- Flatten to feature vectors ---
        Feat_tr = X_train.reshape(X_train.shape[0], -1)
        Feat_va = X_val.reshape(X_val.shape[0], -1)
        Feat_fi = X_fin.reshape(X_fin.shape[0], -1)

        # --- Scale on TRAIN only ---
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Feat_tr)
        Xv = scaler.transform(Feat_va)
        Xte = scaler.transform(Feat_fi)

        if use_pca_whiten:
            pca = PCA(whiten=True, svd_solver='auto', random_state=42)
            Ztr = pca.fit_transform(Xtr)
            Zv = pca.transform(Xv)
            Zte = pca.transform(Xte)
        else:
            Ztr, Zv, Zte = Xtr, Xv, Xte

        if enable_cp_oc_svm:
            with peak_ram(prefix=f"OCSVM", label=f"R={'NA'}", interval=0.02) as m:
                one_class_svm(Ztr, Zv, Zte)

        if enable_cp_autoencoder:
            with peak_ram(prefix=f"AE", label=f"R={'NA'}", interval=0.02) as m:
                autoencoder(Ztr, Zv, Zte)

        if enable_cp_isolation_forest:
            with peak_ram(prefix=f"IF", label=f"R={'NA'}", interval=0.02) as m:
                accuracy = isolation_forest_anomaly(Ztr, Zv, Zte)


if use_predefined_rank:
    print('Predefined rank')
    for split_seed in {1,2,3,5,8,13,21}:
        print('Split seed:', split_seed)

        # Entry (reads once, then passes data to pipelines)
        data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42, split_seed=split_seed)
        X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

        if enable_pca_oc_svm:
            rank = 380
            PCA_WHITEN = True  # set False to skip whitening (uses your scaler)
            RANDOM_SEED = 42
            with peak_ram(prefix="PCA+OCSVM", label=f"rank={rank}", interval=0.02) as m:
                # -------- PCA + OC-SVM pipeline --------
                # Flatten
                Xtr = _flatten(X_train)
                Xva = _flatten(X_val)
                Xte = _flatten(X_fin)

                # Standardize BEFORE PCA (important for RBF kernels downstream)
                scaler0 = StandardScaler()
                Xtr_s = scaler0.fit_transform(Xtr)
                Xva_s = scaler0.transform(Xva)
                Xte_s = scaler0.transform(Xte)

                # Fit PCA on TRAIN only; transform VAL/FINAL
                pca = PCA(n_components=rank, whiten=PCA_WHITEN, svd_solver='auto', random_state=RANDOM_SEED)
                Htr = pca.fit_transform(Xtr_s)
                Hva = pca.transform(Xva_s)
                Hte = pca.transform(Xte_s)

                # If NOT whitening, scale the PCA space (OC-SVM is scale sensitive)
                if not PCA_WHITEN:
                    post_scaler = StandardScaler(with_mean=True, with_std=True)
                    Htr = post_scaler.fit_transform(Htr)
                    Hva = post_scaler.transform(Hva)
                    Hte = post_scaler.transform(Hte)

                best_score_tuple, best_model, best_params, best_aux_print = \
                    OC_SVM('PCA', Htr, Hva, Hte)

                evaluate_OC_SVM('PCA', rank, Hte, y_fin, best_score_tuple, best_model, best_params, best_aux_print)


        if enable_pca_autoencoder:
            rank = 375
            factor = 1
            bottleneck = 16

            PCA_WHITEN = True  # set False to skip whitening (uses your scaler)
            RANDOM_SEED = 42
            with peak_ram(prefix="PCA+AE", label=f"rank={rank}", interval=0.02) as m:
                # -------- PCA + OC-SVM pipeline --------
                # Flatten
                Xtr = _flatten(X_train)
                Xva = _flatten(X_val)
                Xte = _flatten(X_fin)

                # Standardize BEFORE PCA (important for RBF kernels downstream)
                scaler0 = StandardScaler()
                Xtr_s = scaler0.fit_transform(Xtr)
                Xva_s = scaler0.transform(Xva)
                Xte_s = scaler0.transform(Xte)

                # Fit PCA on TRAIN only; transform VAL/FINAL
                pca = PCA(n_components=rank, whiten=PCA_WHITEN, svd_solver='auto', random_state=RANDOM_SEED)
                Htr = pca.fit_transform(Xtr_s)
                Hva = pca.transform(Xva_s)
                Hte = pca.transform(Xte_s)

                # If NOT whitening, scale the PCA space (OC-SVM is scale sensitive)
                if not PCA_WHITEN:
                    post_scaler = StandardScaler(with_mean=True, with_std=True)
                    Htr = post_scaler.fit_transform(Htr)
                    Hva = post_scaler.transform(Hva)
                    Hte = post_scaler.transform(Hte)

                err_va, autoencoder, run_time = neural_network_autoencoder(factor, bottleneck, Htr, Hva, Hte)

                evaluate_AE('PCA', rank, Hte, y_fin, autoencoder)


        if enable_pca_isolation_forest:
            rank = 300
            PCA_WHITEN = True  # set False to skip whitening (uses your scaler)
            RANDOM_SEED = 42
            with peak_ram(prefix="PCA+IF", label=f"rank={rank}", interval=0.02) as m:
                # -------- PCA + OC-SVM pipeline --------
                # Flatten
                Xtr = _flatten(X_train)
                Xva = _flatten(X_val)
                Xte = _flatten(X_fin)

                # Standardize BEFORE PCA (important for RBF kernels downstream)
                scaler0 = StandardScaler()
                Xtr_s = scaler0.fit_transform(Xtr)
                Xva_s = scaler0.transform(Xva)
                Xte_s = scaler0.transform(Xte)

                # Fit PCA on TRAIN only; transform VAL/FINAL
                pca = PCA(n_components=rank, whiten=PCA_WHITEN, svd_solver='auto', random_state=RANDOM_SEED)
                Htr = pca.fit_transform(Xtr_s)
                Hva = pca.transform(Xva_s)
                Hte = pca.transform(Xte_s)

                # If NOT whitening, scale the PCA space (OC-SVM is scale sensitive)
                if not PCA_WHITEN:
                    post_scaler = StandardScaler(with_mean=True, with_std=True)
                    Htr = post_scaler.fit_transform(Htr)
                    Hva = post_scaler.transform(Hva)
                    Hte = post_scaler.transform(Hte)

                best_if, best_obj, thr, best_params = isolation_forest('PCA', Htr_w=Htr, Hval_w=Hva, Hfin_w=Hte)

                evaluate_IF('PCA', rank, Hte, y_fin, best_if, best_obj, thr, best_params)


        if enable_cp_oc_svm:
            # Fixed rank:
            rank = 10
            with peak_ram(prefix=f"CP+OCSVM", label=f"R={rank}", interval=0.02) as m:
                # Global CP fit + project
                (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
                    X_train, X_val, X_fin, rank,
                    random_state=42
                )

                if RUN_CP_VISUALIZATION:
                    print('raw tiles')
                    show_raw_tiles_grayscale(X_train, idxs=[0, 1, 2], bands=(0, 1, 2, 3, 4, 5))
                    print('visualize_cp_reconstruction')
                    visualize_cp_reconstruction(A, B, C, H_train, idxs=[0, 1, 2], bands=(0, 1, 2, 3, 4, 5))

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(H_train)
                Hval_s = scaler.transform(H_val)
                Hfin_s = scaler.transform(H_fin)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                best_score_tuple, best_model, best_params, best_aux_print = \
                    OC_SVM('CP', Htr_w, Hval_w, Hfin_w)

                evaluate_OC_SVM('CP', rank, Hfin_w, y_fin, best_score_tuple, best_model, best_params, best_aux_print)


        if enable_cp_autoencoder:
            # Fixed rank:
            rank = 10
            factor = 2
            bottleneck = 32
            with peak_ram(prefix=f"CP+AE", label=f"R={rank}", interval=0.02) as m:
                # Global CP fit + project
                (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
                    X_train, X_val, X_fin, rank,
                    random_state=42
                )

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(H_train)
                Hval_s = scaler.transform(H_val)
                Hfin_s = scaler.transform(H_fin)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                sum_err_va, autoencoder, run_time = neural_network_autoencoder(factor, bottleneck, Htr_w, Hval_w, Hfin_w)

                evaluate_AE('CP', rank, Hfin_w, y_fin, autoencoder)

        if enable_cp_isolation_forest:
            # Fixed rank:
            rank = 115
            with peak_ram(prefix=f"CP+IF", label=f"R={rank}", interval=0.02) as m:
                # Global CP fit + project
                (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
                    X_train, X_val, X_fin, rank,
                    random_state=42
                )

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(H_train)
                Hval_s = scaler.transform(H_val)
                Hfin_s = scaler.transform(H_fin)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                best_if, best_obj, thr, best_params = isolation_forest('CP', Htr_w=Htr_w, Hval_w=Hval_w, Hfin_w=Hfin_w)

                evaluate_IF('CP', rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params)


        if enable_tucker_oc_svm:
            rank = (5, 5, 5)
            with peak_ram(prefix=f"Tucker+OCSVM", label=f"R={rank}", interval=0.02) as m:
                Z_tr, Z_va, Z_fi = None, None, None
                # Tucker decompositions
                n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
                decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
                decomp_va = buildTensor(X_val, rank, n_va, isTuckerDecomposition=True)
                decomp_fi = buildTensor(X_fin, rank, n_fi, isTuckerDecomposition=True)

                # Feature extraction + scaling (fit on TRAIN only)
                Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)

                if RUN_TUCKER_VISUALIZATION:
                    # Show 3 specific Tucker reconstructions across all 6 bands
                    visualize_tucker_reconstruction_per_tile(decomp_tr, idxs=[0, 1, 2], bands=(0, 1, 2, 3, 4, 5))

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(Feat_tr)
                Hval_s = scaler.transform(Feat_va)
                Hfin_s = scaler.transform(Feat_fi)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                best_tuple, best_model, best_params, best_aux = OC_SVM('Tucker', Htr_w, Hval_w, Hfin_w)

                evaluate_OC_SVM('Tucker', rank, Hfin_w, y_fin, best_tuple, best_model, best_params,
                                best_aux)

        if enable_tucker_autoencoder:
            rank = (5, 5, 5)
            factor = 2
            bottleneck = 16
            with peak_ram(prefix=f"Tucker+AE", label=f"R={rank}", interval=0.02) as m:
                Z_tr, Z_va, Z_fi = None, None, None
                # Tucker decompositions
                n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
                decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
                decomp_va = buildTensor(X_val, rank, n_va, isTuckerDecomposition=True)
                decomp_fi = buildTensor(X_fin, rank, n_fi, isTuckerDecomposition=True)

                # Feature extraction + scaling (fit on TRAIN only)
                Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(Feat_tr)
                Hval_s = scaler.transform(Feat_va)
                Hfin_s = scaler.transform(Feat_fi)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                sum_err_va, autoencoder, run_time = neural_network_autoencoder(factor, bottleneck, Htr_w, Hval_w,
                                                                               Hfin_w)

                evaluate_AE('Tucker', rank, Hfin_w, y_fin, autoencoder)


        if enable_tucker_isolation_forest:
            rank = (32, 32, 16)
            with peak_ram(prefix=f"Tucker+IF", label=f"R={rank}", interval=0.02) as m:
                Z_tr, Z_va, Z_fi = None, None, None
                # Tucker decompositions
                n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
                decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
                decomp_va = buildTensor(X_val, rank, n_va, isTuckerDecomposition=True)
                decomp_fi = buildTensor(X_fin, rank, n_fi, isTuckerDecomposition=True)

                # Feature extraction + scaling (fit on TRAIN only)
                Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)
                Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True,
                                          feature_mode=TUCKER_FEATURE_MODE)

                # Scale
                scaler = StandardScaler()
                Htr_s = scaler.fit_transform(Feat_tr)
                Hval_s = scaler.transform(Feat_va)
                Hfin_s = scaler.transform(Feat_fi)

                # Feature pathway: full H (optionally PCA-whiten)
                if use_pca_whiten:
                    pca = PCA(whiten=True, svd_solver='auto', random_state=42)
                    Htr_w = pca.fit_transform(Htr_s)
                    Hval_w = pca.transform(Hval_s)
                    Hfin_w = pca.transform(Hfin_s)
                else:
                    Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

                best_if, best_obj, thr, best_params = isolation_forest('Tucker', Htr_w, Hval_w, Hfin_w)

                evaluate_IF('Tucker', rank, Hfin_w, y_fin, best_if, best_obj, thr, best_params)
