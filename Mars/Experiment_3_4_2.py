# Imports
import os
import time
import random
import torch
import warnings
import numpy as np
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

random.seed(1)

# Paths & toggles
#train_data        = "Data/Full/train_typical"        # typical only
#validation_data   = "Data/Full/validation_typical"   # typical only
#test_typical_data = "Data/Full/test_typical" # typical
#test_anomaly_data = "Data/Full/test_novel/all"   # novel

train_data        = "Data/Reduced/set_1/train"        # typical only
validation_data   = "Data/Reduced/set_1/validation"   # typical only
test_typical_data = "Data/Reduced/set_1/test_typical" # typical
#test_anomaly_data = "Data/Reduced/set_2/test_novel"   # novel
#test_anomaly_data = "Data/Full/test_novel/bedrock"   # novel
#test_anomaly_data = "Data/Full/test_novel/broken-rock"   # novel
#test_anomaly_data = "Data/Full/test_novel/drill-hole"   # novel
#test_anomaly_data = "Data/Full/test_novel/drt"   # novel
#test_anomaly_data = "Data/Full/test_novel/dump-pile"   # novel
#test_anomaly_data = "Data/Full/test_novel/float"   # novel
#test_anomaly_data = "Data/Full/test_novel/meteorite"   # novel
#test_anomaly_data = "Data/Full/test_novel/scuff"   # novel
test_anomaly_data = "Data/Full/test_novel/veins"   # novel

use_predefined_rank = True
enable_tucker_oc_svm = False
enable_tucker_autoencoder = False
enable_tucker_isolation_forest = False
enable_cp_oc_svm = True
enable_cp_autoencoder = True
enable_cp_isolation_forest = True

no_decomposition = True  # set to False to run raw pixel models
RUN_VISUALIZATION = False

# Optional: standardize bands using TRAIN stats
USE_BAND_STANDARDIZE = True

# Dataset reduction controls
REDUCE_DATASETS = True
REDUCE_TRAIN_N = 1500
REDUCE_VAL_N = 200
REDUCE_TEST_TYP_N = 30
REDUCE_TEST_ANO_N = 30
REDUCE_SEED = 1
VAL_FRACTION = 0.5  # only used if no separate validation dir

# TensorLy backend + device toggles
TL_BACKEND = "pytorch"   # change to "numpy" to force CPU
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

def show_raw_tile(X, idx=0, bands=(0,1,2,3,4,5), title="Raw tile"):
    Xo = np.asarray(X)[idx]  # (64,64,6)
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.5))
    for i, b in enumerate(bands):
        ax = axes[i]
        ax.imshow(Xo[:, :, b], interpolation="nearest")
        ax.set_title(f"raw band {b}")
        ax.axis("off")
    fig.suptitle(f"{title} (idx={idx})")
    plt.tight_layout(); plt.show()

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

def visualize_cp_reconstruction(A, B, C, H, X_ref=None, idx=0, bands=(0,1,2,3,4,5), title_prefix=""):
    """
    Reconstruct tile `idx` from H and optionally show original beside it.
    """
    h = np.asarray(H)[idx]
    Xhat = cp_reconstruct_tile(A, B, C, h)

    n = len(bands)
    plt.figure(figsize=(3*n, 6))
    for i, b in enumerate(bands):
        # Top: original (if provided) or reconstruction
        ax = plt.subplot(2, n, i+1)
        if X_ref is not None:
            ax.imshow(np.asarray(X_ref)[idx, :, :, b], interpolation='nearest')
            ax.set_title(f"orig band {b}")
        else:
            ax.imshow(Xhat[:, :, b], interpolation='nearest')
            ax.set_title(f"recon band {b}")
        ax.axis('off')

        # Bottom: reconstruction
        ax = plt.subplot(2, n, n+i+1)
        ax.imshow(Xhat[:, :, b], interpolation='nearest')
        ax.set_title(f"recon band {b}")
        ax.axis('off')

    supt = f"{title_prefix} CP reconstruction (idx={idx})"
    plt.suptitle(supt)
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
    decomp_list, X_ref, idx=0, bands=(0, 1, 2),
    title_prefix="Tucker (per-tile)", save_path=None
):
    """
    Show ORIGINAL vs RECONSTRUCTION side-by-side per selected band
    for a *per-tile* Tucker decomposition result.

    Parameters
    ----------
    decomp_list : list[(core,[U1,U2,U3])]
        Output of buildTensor(..., isTuckerDecomposition=True).
    X_ref : np.ndarray
        Stack for that split, shape (N,64,64,6).
    idx : int
        Tile index to visualize.
    bands : tuple[int]
        Bands to display.
    title_prefix : str
        Title prefix (e.g., 'Tucker rank=(5,5,3)').
    save_path : str or None
        If provided, save the figure to this path.
    """
    core, U1, U2, U3 = _unpack_tucker_from_decomp(decomp_list[idx])
    Xhat = tucker_reconstruct_tile(core, U1, U2, U3)

    n_b = len(bands)
    n_cols = 2  # orig | recon
    fig, axes = plt.subplots(nrows=n_b, ncols=n_cols,
                             figsize=(4.2 * n_cols, 3.8 * n_b), squeeze=False)

    for i, b in enumerate(bands):
        Xo = np.asarray(X_ref)[idx, :, :, b]
        Xr = Xhat[:, :, b]
        # robust shared scaling per band
        lo = np.percentile([Xo.min(), Xr.min()], 1)
        hi = np.percentile([Xo.max(), Xr.max()], 99)

        ax = axes[i, 0]
        ax.imshow(Xo, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"orig (band {b})"); ax.axis("off")

        ax = axes[i, 1]
        ax.imshow(Xr, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"recon (band {b})"); ax.axis("off")

    fig.suptitle(f"{title_prefix} reconstruction (tile idx={idx})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160)
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
def prepare_data_once(val_fraction=VAL_FRACTION, random_state=42):
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
        random_state=REDUCE_SEED
    )

    # TEST pool (typical + anomalies)
    X_pool, y_pool = readData_test(
        test_typical_data, test_anomaly_data,
        max_typical=(REDUCE_TEST_TYP_N if REDUCE_DATASETS else None),
        max_anomaly=(REDUCE_TEST_ANO_N if REDUCE_DATASETS else None),
        random_state=REDUCE_SEED
    )
    y_pool = np.asarray(y_pool, int)

    # VALIDATION
    if _dir_has_npy(validation_data):
        X_val, _ = readData(
            validation_data,
            max_files=(REDUCE_VAL_N if REDUCE_DATASETS else None),
            random_state=REDUCE_SEED + 7
        )
        y_val = None
        X_fin, y_fin = X_pool, y_pool
        print(f"Validation: using separate directory {validation_data} (N={X_val.shape[0]})")
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

# CP + OC-SVM using preloaded data
def parafac_OC_SVM(rank, data_bundle,
                   displayConfusionMatrix=False, use_pca_whiten=True,
                   random_state=42):
    """
    Pipeline:
      1) Standardize using TRAIN stats (optional).
      2) Fit a single global CP basis on TRAIN (optionally subsampled).
      3) Project VAL and FINAL onto the basis.
      4) Select OC-SVM params on VAL (AUC if labels present, else one-class).
      5) Evaluate on FINAL.
    """
    start_time = time.time()

    # Common split & standardization
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

    # Global CP fit + project
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state
    )

    if RUN_VISUALIZATION:
        print('raw tiles')
        show_raw_tile(X_train, idx=0)
        print('visualize_cp_reconstruction')
        visualize_cp_reconstruction(A, B, C, H_train, X_ref=X_train, idx=0)

    # Scale
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    # Feature pathway: full H (optionally PCA-whiten)
    if use_pca_whiten:
        pca   = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Htr_w  = pca.fit_transform(Htr_s)
        Hval_w = pca.transform(Hval_s)
        Hfin_w = pca.transform(Hfin_s)
    else:
        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

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

    print(f"CP+OCSVM (VAL one-class) rank {rank} chose {best_params} ({best_aux_print})")

    print("Elapsed:", round(time.time() - start_time, 2), "s")
    return best_score_tuple, best_model, Hfin_w, y_fin, best_params, best_aux_print


def ocsvm_only(
    data_bundle,
    displayConfusionMatrix=False,
    use_pca_whiten=False,
    random_state=42 ):
    """
    OC-SVM on raw (flattened) tiles, no CP.
    - Expects `data_bundle` from your prepare_data_once(...) path.
    - VAL is typical-only → choose params by minimizing FP@0 (ties: P95, mean).
    - No robust fallbacks.
    - Optional PCA (dim reduction). Set do_pca=False to disable PCA entirely.
    Returns: (accuracy, auc) on FINAL.
    """
    start_time = time.time()

    # --- Split (respect your band standardization toggle) ---
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(
        data_bundle, standardize=USE_BAND_STANDARDIZE
    )

    # --- Flatten to feature vectors ---
    Feat_tr = X_train.reshape(X_train.shape[0], -1)
    Feat_va = X_val.reshape(X_val.shape[0], -1)
    Feat_fi = X_fin.reshape(X_fin.shape[0], -1)

    # --- Scale on TRAIN only ---
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Feat_tr)
    Xv  = scaler.transform(Feat_va)
    Xte = scaler.transform(Feat_fi)

    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Ztr = pca.fit_transform(Xtr)
        Zv = pca.transform(Xv)
        Zte = pca.transform(Xte)
    else:
        Ztr, Zv, Zte = Xtr, Xv, Xte


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


def one_class_svm():
    print("One-class SVM (raw pixels)")
    acc, auc = ocsvm_only(data_bundle, displayConfusionMatrix=False, use_pca_whiten=False)
    print("One-class SVM best accuracy:", acc, "auc:", auc)

# Rank search now reuses the preloaded bundle
def cp_rank_search_one_class_svm(data_bundle):
    print("CP rank search (One-class SVM)")

    rank_best_score_tuple = None
    rank_best_model = None
    rank_best_rank = None
    rank_H_fin = None
    rank_best_params = None
    rank_best_aux_print = None

    startRank = 10; endRank = 385; step = 5  # tighter range for speed
    for rank in range(startRank, endRank, step):
        print("Rank:", rank)
        best_score_tuple, best_model, Hfin_w, y_fin, best_params, best_aux_print = parafac_OC_SVM(rank, data_bundle, use_pca_whiten=True)
        if (rank_best_score_tuple is None or best_score_tuple < rank_best_score_tuple):
            rank_best_score_tuple = best_score_tuple
            rank_best_model = best_model
            rank_best_rank = rank
            rank_H_fin = Hfin_w
            rank_best_params = best_params
            rank_best_aux_print = best_aux_print

            s_fin = -rank_best_model.decision_function(rank_H_fin).ravel()
            auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
            print(f"CP+OCSVM Intermediate result RANK={rank_best_rank} AUC={auc_fin} | ACC={acc_opt} "
                  f"| param=({rank_best_params}) | aux={rank_best_aux_print}" )

    # FINAL evaluation
    s_fin = -rank_best_model.decision_function(rank_H_fin).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"CP+OCSVM Final result RANK={rank_best_rank} AUC={auc_fin} | ACC={acc_opt} "
          f"| param=({rank_best_params}) | aux={rank_best_aux_print}" )


#
# Tucker with OC-SVM (shared data path)
#
def tucker_one_class_svm(rank, data_bundle, displayConfusionMatrix=False,
                         random_state=42, val_fraction=0.5, feature_mode=TUCKER_FEATURE_MODE):
    """
    Tucker + OC-SVM using the common read/standardize path.
    feature_mode: "both", "core", or "factors" (for Tucker features).
    """
    # Common split & standardization
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

    # Tucker decompositions
    n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
    start_time = time.time()
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_va = buildTensor(X_val,   rank, n_va, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)
    print(f"Decomposition time: {time.time() - start_time:.2f} seconds | features={feature_mode}")

    if RUN_VISUALIZATION:
        visualize_tucker_reconstruction_per_tile(
            decomp_tr, X_train, idx=0, bands=(0, 1, 2, 3, 4, 5),
            title_prefix=f"Tucker rank={tuple(rank)}",
            save_path=f"viz/tucker_recon_rank{tuple(rank)}_tile0.png"
        )

    # Feature extraction + scaling (fit on TRAIN only)
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True, feature_mode=feature_mode)
    Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True, feature_mode=feature_mode)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True, feature_mode=feature_mode)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_va = scaler.transform(Feat_va)
    Z_fi = scaler.transform(Feat_fi)

    # Hyperparameter search on VAL
    d = Z_tr.shape[1]
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(d) + ["scale", "auto"],
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    best_tuple = None
    best_model = None
    best_params = None
    best_aux = ""

    for p in ParameterGrid(param_grid):
        m = OneClassSVM(**p).fit(Z_tr)
        s_val = -m.decision_function(Z_va).ravel()  # anomaly-positive scores
        if not np.all(np.isfinite(s_val)):
            continue

        fp  = float((s_val >= 0.0).mean())
        p95 = float(np.percentile(s_val, 95))
        mean_s = float(np.mean(s_val))
        obj = (fp, p95, mean_s)
        aux = f"FP={fp:.3f}, P95={p95:.4f}, mean={mean_s:.4f}"

        if (best_tuple is None) or (obj < best_tuple):
            best_tuple = obj
            best_model = m
            best_params = dict(p)
            best_aux = aux

    sel_mode = "one-class"
    print(f"Tucker+OCSVM ({feature_mode}, VAL {sel_mode}) chose {best_params} ({best_aux})")

    return best_model, best_tuple, Z_fi, y_fin, best_params, best_aux


def tucker_rank_search_one_class_svm(data_bundle):
    """
    Grid over Tucker rank triples; choose the rank with highest FINAL accuracy.
    """
    print("Tucker rank search (One-class SVM)")
    rankSet = sorted({5, 16, 32, 64})
    rank_best_model = None
    rank_best_tuple = None
    rank_Z_fi = None
    rank_y_fin = None
    rank_best_rank = None
    rank_best_params= None
    rank_best_aux= None
    for i in rankSet:
        for j in rankSet:
            for k in sorted({5, 16}):
                r = (i, j, k)
                print("Rank:", i, j, k)
                best_model, best_tuple, Z_fi, y_fin, best_params, best_aux = tucker_one_class_svm(r, data_bundle, feature_mode=TUCKER_FEATURE_MODE)
                if (rank_best_tuple is None) or (best_tuple < rank_best_tuple):
                    rank_best_tuple = best_tuple
                    rank_best_model = best_model
                    rank_Z_fi = Z_fi
                    rank_y_fin = y_fin
                    rank_best_rank = r
                    rank_best_params = best_params
                    rank_best_aux = best_aux

                    # Intermediate evaluation
                    s_fin = -rank_best_model.decision_function(rank_Z_fi).ravel()
                    auc_fin = manual_auc(rank_y_fin, s_fin, positive_label=-1)
                    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, s_fin, positive_label=-1)
                    print(f"Tucker+OCSVM Intermediate result RANK={rank_best_rank} AUC={auc_fin} | ACC={acc_opt} "
                          f"| param={rank_best_params} | aux={rank_best_aux}")

    # FINAL evaluation
    s_fin = -rank_best_model.decision_function(rank_Z_fi).ravel()
    auc_fin = manual_auc(rank_y_fin, s_fin, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, s_fin, positive_label=-1)
    print(f"Tucker+OCSVM Final result RANK={rank_best_rank} AUC={auc_fin} | ACC={acc_opt} "
          f"| param={rank_best_params} | aux={rank_best_aux}")


#
# CP with Autoencoder (uses common CP fit/proj + common data)
#
def parafac_autoencoder(rank, factor, bottleneck, data_bundle,
                        displayConfusionMatrix=False,
                        random_state=42, cp_basis_max_train_samples=None, use_pca_whiten=False):
    """
    CP+Autoencoder using a single global CP basis fit on TRAIN, then projecting
    VAL/FINAL onto that basis.
    Prints FINAL AUC and the max-accuracy threshold metrics.
    """
    # Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Global CP -> H matrices
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state
    )

    # Scale features (and optional PCA whitening) fit on TRAIN only
    scaler = StandardScaler()
    Htr_s = scaler.fit_transform(H_train)
    Hva_s = scaler.transform(H_val)
    Hfi_s = scaler.transform(H_fin)

    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Z_tr  = pca.fit_transform(Htr_s)
        Z_va  = pca.transform(Hva_s)
        Z_fi  = pca.transform(Hfi_s)
    else:
        Z_tr, Z_va, Z_fi = Htr_s, Hva_s, Hfi_s

    # Define Autoencoder (simple MLP bottleneck)
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

    # Use separate VAL (typical-only) for early stopping and threshold
    autoencoder.fit(Z_tr, Z_tr, epochs=10, batch_size=32,
                    validation_data=(Z_va, Z_va),
                    callbacks=[es], verbose=0)

    # Threshold from VAL
    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean(np.square(Z_va - recon_va), axis=1)
    threshold = np.percentile(err_va, 95)

    print(f'Train/Val err rank={rank} sum={np.sum(err_va)} err mean={np.mean(err_va)} threshold={threshold}')
    return np.sum(err_va), autoencoder, Z_fi, y_fin


def autoencoder_anomaly(data_bundle, factor, bottleneck, use_pca_whiten=True, random_state=42):
    """
    Raw-pixel autoencoder using the shared data path and validation split.
    """
    # Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Flatten + scale (fit on TRAIN)
    n_tr = X_train.shape[0]
    scaler = StandardScaler()
    Htr_s = scaler.fit_transform(X_train.reshape(n_tr, -1))
    # Validation
    n_va = X_val.shape[0]
    Hva_s = scaler.transform(X_val.reshape(n_va, -1))
    # TEST/FINAL
    n_te = X_fin.shape[0]
    Hfi_s = scaler.transform(X_fin.reshape(n_te, -1))

    # --- optional PCA whitening on H ---
    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Z_tr = pca.fit_transform(Htr_s)
        Z_va = pca.transform(Hva_s)
        Z_fi = pca.transform(Hfi_s)
    else:
        Z_tr, Z_va, Z_fi = Htr_s, Hva_s, Hfi_s

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


def autoencoder(data_bundle, displayConfusionMatrix=False, sweep_factors=(1, 2, 3), sweep_bottlenecks=(16, 32, 64)):
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
            print("Factor:", factor, "Bottleneck:", bottleneck)
            err_va, autoencoder, Z_fi, y_fin = autoencoder_anomaly(data_bundle, factor, bottleneck)

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
                    print(f"CP+AE (global) Intermediate result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")

    # Score FINAL
    recon_fi = rank_autoencoder.predict(rank_Z_fi, verbose=0)
    err_fi = np.mean(np.square(rank_Z_fi - recon_fi), axis=1)  # anomaly-positive scores
    auc_fin = manual_auc(rank_y_fin, err_fi, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(rank_y_fin, err_fi, positive_label=-1)
    print(f"CP+AE (global) Final result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")


#
# Tucker with Autoencoder (shared path)
#
def tucker_neural_network_autoencoder(rank, factor, bottleneck, data_bundle,
                                      displayConfusionMatrix=False, feature_mode=TUCKER_FEATURE_MODE):
    """
    Tucker features -> simple autoencoder. Prints AUC and max-accuracy threshold.
    """
    # Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Tucker decomposition per split
    n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_va = buildTensor(X_val,   rank, n_va, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)

    # Extract and normalize features (fit scaler on TRAIN only)
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True, feature_mode=feature_mode)
    Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True, feature_mode=feature_mode)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True, feature_mode=feature_mode)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_va = scaler.transform(Feat_va)
    Z_fi = scaler.transform(Feat_fi)

    1# Define the autoencoder model
    input_dim = Z_tr.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)

    # Decoder
    decoder = Dense(128 * factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback (use VAL split)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(Z_tr, Z_tr, epochs=10, batch_size=32,
                    validation_data=(Z_va, Z_va),
                    callbacks=[early_stopping], verbose=0)

    # Threshold from VAL (typical-only directory, if provided)
    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean(np.square(Z_va - recon_va), axis=1)
    threshold = np.percentile(err_va, 95)

    print(f'Train/Val err rank={rank} sum={np.sum(err_va)} err mean={np.mean(err_va)} threshold={threshold}')
    return np.sum(err_va), autoencoder, Z_fi, y_fin


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
                            print(f"Tucker+AE Intermediate result Rank={best_rank} AUC={auc_fin} ACC={acc_opt}")


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

#
# CP (global basis) + Isolation Forest
#
def parafac_isolation_forest(rank, data_bundle,
                             displayConfusionMatrix=False,
                             random_state=42,
                             cp_basis_max_train_samples=None,
                             use_pca_whiten=False):
    """
    CP features via a global CP basis -> project to H, then IsolationForest.

    If USE_VAL_FOR_IF is True and the validation split is typical-only (y_val is None),
    we (a) select hyperparameters that minimize the 95th percentile of anomaly scores on VAL,
    and (b) calibrate a threshold on VAL at the (1-VAL_FP_TARGET) quantile of scores.

    Otherwise we fall back to TRAIN-only unsupervised selection.
    """
    # --- splits (note we keep y_val this time) ---
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # --- CP basis + projection ---
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state )

    # --- scale (fit on TRAIN only) ---
    scaler = StandardScaler()
    Htr_s = scaler.fit_transform(H_train)
    Hva_s = scaler.transform(H_val)
    Hfi_s = scaler.transform(H_fin)

    # --- optional PCA whitening on H ---
    if use_pca_whiten:
        pca  = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Z_tr = pca.fit_transform(Htr_s)
        Z_va = pca.transform(Hva_s)
        Z_fi = pca.transform(Hfi_s)
    else:
        Z_tr, Z_va, Z_fi = Htr_s, Hva_s, Hfi_s

    # --- grid ---
    warnings.filterwarnings('ignore', category=UserWarning)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples':  [0.5, 0.75, 1.0],
        'contamination':[0.05, 0.10, 0.20],  # only used by .predict default; we ignore at inference
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

    return best_if, best_obj, thr, Z_fi, y_fin, best_params


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
            print(f"CP+IF Intermediate result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
                  f"| params={rank_best_param} "
                  f"| thr={rank_best_thr:.6f} "
                  f"| target_FP={VAL_FP_TARGET:.3f}")

    # --- FINAL scoring ---
    s_fi = -rank_best_if.score_samples(rank_Z_fi)  # anomaly-positive
    preds = np.where(s_fi >= rank_best_thr, -1, 1)
    acc = float(np.mean(preds == rank_y_fin))
    auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
    print(f"CP+IF Final result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
          f"| params={rank_best_param} "
          f"| thr={rank_best_thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")


# Raw-pixel IsolationForest (no decomposition), using shared data path
def isolation_forest_anomaly(data_bundle, use_pca_whiten=True, random_state=42):
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # --- Flatten to feature vectors ---
    Feat_tr = X_train.reshape(X_train.shape[0], -1)
    Feat_va = X_val.reshape(X_val.shape[0], -1)
    Feat_fi = X_fin.reshape(X_fin.shape[0], -1)

    # --- scale (fit on TRAIN only) ---
    scaler = StandardScaler()
    Htr_s = scaler.fit_transform(Feat_tr)
    Hva_s = scaler.transform(Feat_va)
    Hfi_s = scaler.transform(Feat_fi)

    # --- optional PCA whitening on H ---
    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Z_tr = pca.fit_transform(Htr_s)
        Z_va = pca.transform(Hva_s)
        Z_fi = pca.transform(Hfi_s)
    else:
        Z_tr, Z_va, Z_fi = Htr_s, Hva_s, Hfi_s

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
def tucker_isolation_forests(rank, data_bundle, displayConfusionMatrix=False, random_state=42,
                             feature_mode=TUCKER_FEATURE_MODE):
    """
    Tucker features -> IsolationForest.

    With typical-only VAL (y_val is None) and USE_VAL_FOR_IF=True,
    we select IF hyperparameters that minimize P95 of VAL scores and set the
    operating threshold from the VAL score quantile corresponding to VAL_FP_TARGET.
    """
    # --- splits (keep y_val) ---
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # --- Tucker decomposition (TRAIN/FINAL); we don't need factors for VAL if using only scores ---
    n_tr, n_fi = X_train.shape[0], X_fin.shape[0]
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)

    # --- features + scaling (fit on TRAIN) ---
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True, feature_mode=feature_mode)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True, feature_mode=feature_mode)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_fi = scaler.transform(Feat_fi)

    # Build VAL features too (needed for VAL-based selection/threshold)
    decomp_va = buildTensor(X_val, rank, X_val.shape[0], isTuckerDecomposition=True)
    Feat_va = extractFeatures(decomp_va, X_val.shape[0], isTuckerDecomposition=True, feature_mode=feature_mode)
    Z_va = scaler.transform(Feat_va)

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

    return best_if, best_obj, thr, Z_fi, y_fin, best_params


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
                    print(f"Tucker+IF Intermediate result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
                          f"| params={rank_best_param} "
                          f"| thr={rank_best_thr:.6f} "
                          f"| target_FP={VAL_FP_TARGET:.3f}")

    # --- FINAL scoring ---
    s_fi = -rank_best_if.score_samples(rank_Z_fi)
    preds = np.where(s_fi >= rank_best_thr, -1, 1)
    acc = float(np.mean(preds == rank_y_fin))
    auc_fin = manual_auc(rank_y_fin, s_fi, positive_label=-1)
    print(f"Tucker+IF Final result rank={rank_best_rank} AUC={auc_fin} ACC={acc} obj={rank_best_obj}"
          f"| params={rank_best_param} "
          f"| thr={rank_best_thr:.6f} "
          f"| target_FP={VAL_FP_TARGET:.3f}")


# Entry (reads once, then passes data to pipelines)
data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42)

if enable_cp_oc_svm:
    if no_decomposition:
        one_class_svm()
    else:
        if use_predefined_rank == False:
            cp_rank_search_one_class_svm(data_bundle)
        else:
            rank = 120
            best_score_tuple, best_model, Hfin_w, y_fin, best_params, best_aux_print = parafac_OC_SVM(rank, data_bundle, use_pca_whiten=True)

            # FINAL evaluation
            s_fin = -best_model.decision_function(Hfin_w).ravel()
            auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
            print(f"CP+OCSVM Final result RANK={rank} AUC={auc_fin} | ACC={acc_opt} "
                  f"| param=({best_params}) | aux={best_aux_print}")

if enable_tucker_oc_svm:
    if use_predefined_rank == False:
        tucker_rank_search_one_class_svm(data_bundle)
    else:
        print("Running Tucker OC-SVM at a fixed rank")
        rank = (32, 32, 16)
        best_model, best_tuple, Z_fi, y_fin, best_params, best_aux = tucker_one_class_svm(rank, data_bundle, True, feature_mode=TUCKER_FEATURE_MODE)

        # FINAL evaluation
        s_fin = -best_model.decision_function(Z_fi).ravel()
        auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
        th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
        print(f"Tucker+OCSVM Final result RANK={rank} AUC={auc_fin} | ACC={acc_opt} "
              f"| param={best_params} | aux={best_aux}")

if enable_cp_autoencoder:
    if no_decomposition:
        autoencoder(data_bundle)
    else:
        if use_predefined_rank == False:
            cp_rank_search_autoencoder(data_bundle)
        else:
            print("Running CP+autoencoder at a fixed rank")
            bestRank = 35
            err_va, autoencoder, Z_fi, y_fin = parafac_autoencoder(bestRank, factor=3, bottleneck=64, data_bundle=data_bundle)

            # Score FINAL
            recon_fi = autoencoder.predict(Z_fi, verbose=0)
            err_fi = np.mean(np.square(Z_fi - recon_fi), axis=1)  # anomaly-positive scores
            auc_fin = manual_auc(y_fin, err_fi, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_fi, positive_label=-1)
            print(f"CP+AE (global) Final result Rank={bestRank} AUC={auc_fin} ACC={acc_opt}")

if enable_tucker_autoencoder:
    if use_predefined_rank == False:
        tucker_rank_search_autoencoder(data_bundle)
    else:
        print("Running Tucker+autoencoder at a fixed rank")
        rank = (64, 32, 5)
        factor = 3
        bottleneck=64
        err_va, autoencoder, Z_fi, y_fin = tucker_neural_network_autoencoder(rank, factor, bottleneck, data_bundle, True, feature_mode=TUCKER_FEATURE_MODE)

        # Predict on FINAL
        recon_fi = autoencoder.predict(Z_fi, verbose=0)
        err_fi = np.mean(np.square(Z_fi - recon_fi), axis=1)  # anomaly-positive scores
        # AUC + max-accuracy threshold (for parity with other strategies)
        auc_fin = manual_auc(y_fin, err_fi, positive_label=-1)
        th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_fi, positive_label=-1)
        print(f"Tucker+AE Final result Rank={rank} AUC={auc_fin} ACC={acc_opt}")

if enable_cp_isolation_forest:
    if no_decomposition:
        print("Isolation Forest (raw pixels)")
        accuracy = isolation_forest_anomaly(data_bundle)
    else:
        if use_predefined_rank == False:
            cp_rank_search_isolation_forest(data_bundle)
        else:
            print("Running CP+Isolation Forest at a fixed rank")
            bestRank = 365
            best_if, best_obj, thr, Z_fi, y_fin, best_params = parafac_isolation_forest(bestRank, data_bundle, True)

            # --- FINAL scoring ---
            s_fi = -best_if.score_samples(Z_fi)  # anomaly-positive
            preds = np.where(s_fi >= thr, -1, 1)
            acc = float(np.mean(preds == y_fin))
            auc_fin = manual_auc(y_fin, s_fi, positive_label=-1)
            print(f"CP+IF Final result rank={bestRank} AUC={auc_fin} ACC={acc} obj={best_obj}"
                  f"| params={best_params} "
                  f"| thr={thr:.6f} "
                  f"| target_FP={VAL_FP_TARGET:.3f}")


if enable_tucker_isolation_forest:
    if use_predefined_rank == False:
        tucker_rank_search_isolation_forest(data_bundle)
    else:
        print("Running Tucker+Isolation Forest at a fixed rank")
        rank = (64, 16, 5)
        best_if, best_obj, thr, Z_fi, y_fin, best_params = tucker_isolation_forests(rank, data_bundle, True, feature_mode=TUCKER_FEATURE_MODE)

        # --- FINAL scoring ---
        s_fi = -best_if.score_samples(Z_fi)
        preds = np.where(s_fi >= thr, -1, 1)
        acc = float(np.mean(preds == y_fin))
        auc_fin = manual_auc(y_fin, s_fi, positive_label=-1)
        print(f"Tucker+IF Final result rank={rank} AUC={auc_fin} ACC={acc} obj={best_obj}"
              f"| params={best_params} "
              f"| thr={thr:.6f} "
              f"| target_FP={VAL_FP_TARGET:.3f}")
