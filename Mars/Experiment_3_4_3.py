# ====== Imports ===============================================================
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# NEW (for the Tucker OC-SVM snippet you provided)
import warnings
from sklearn.model_selection import GridSearchCV

# TensorLy
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.decomposition import parafac as _tl_parafac, tucker as _tl_tucker

# ML utils
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit

# --- NEW: Keras (for autoencoder) ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer  # (not strictly required but safe to have)

random.seed(1)

# ====== Paths & toggles (as in your project) =================================
train_data        = "Data/Reduced/Lean/train"        # typical only
validation_data   = "Data/Reduced/Lean/validation"   # typical only
test_typical_data = "Data/Reduced/Lean/test_typical" # typical
test_anomaly_data = "Data/Reduced/Lean/test_novel"   # novel

use_predefined_rank = False
enable_tucker_oc_svm = False
enable_tucker_autoencoder = False
enable_tucker_isolation_forest = False
enable_cp_oc_svm = True
enable_cp_autoencoder = False
enable_cp_isolation_forest = False

no_decomposition = False  # set to False to run CP-based pipeline

RUN_VISUALIZATION = True

# Optional: standardize bands using TRAIN stats (recommended)
USE_BAND_STANDARDIZE = True

# ====== Dataset reduction controls ===========================================
REDUCE_DATASETS = True
REDUCE_TRAIN_N = 1500
REDUCE_VAL_N = 200
REDUCE_TEST_TYP_N = 200
REDUCE_TEST_ANO_N = 200
REDUCE_SEED = 1  # reproducible subsets
VAL_FRACTION = 0.5 # only used if no separate validation dir

# ====== TensorLy backend + numeric-stability helpers =========================
TL_BACKEND = "pytorch"   # change to "numpy" to force CPU
DEVICE = "cpu"
USE_GPU_CP = True

def _set_tl_backend():
    """Set TensorLy backend; prefer CUDA if pytorch selected and available."""
    global DEVICE
    if TL_BACKEND.lower() == "pytorch":
        try:
            import torch
            tl.set_backend("pytorch")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if DEVICE == "cuda":
                torch.set_num_threads(1)
                print("[tensorly] Backend: pytorch (CUDA)")
            else:
                print("[tensorly] Backend: pytorch (CPU fallback)")
        except Exception as e:
            print(f"[tensorly] PyTorch unavailable ({e}); falling back to numpy.")
            tl.set_backend("numpy")
            DEVICE = "cpu"
            print("[tensorly] Backend: numpy")
    else:
        tl.set_backend("numpy")
        DEVICE = "cpu"
        print("[tensorly] Backend: numpy")

_set_tl_backend()

def _to_backend(x, use_float64=False):
    """Numpy -> backend tensor (contiguous). Use float64 on GPU for stability."""
    arr = np.asarray(x, dtype=np.float64 if (tl.get_backend()=="pytorch" and use_float64) else np.float32, order="C")
    if tl.get_backend() == "pytorch":
        import torch
        return torch.from_numpy(arr).to(DEVICE, non_blocking=True).contiguous()
    else:
        return tl.tensor(arr)

def _to_numpy(x):
    """Backend tensor -> numpy."""
    if tl.get_backend() == "pytorch":
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return tl.to_numpy(x)

def sanity_report(X, name):
    print(f"[sanity] {name}: N={X.shape[0]}, NaNs={np.isnan(X).any()}, Infs={np.isinf(X).any()}")
    if X.size:
        print(f"[sanity] {name}: min={np.nanmin(X):.6f}, max={np.nanmax(X):.6f}")

def clean_tiles(X):
    """One-shot cleaning if standardized tiles contain NaN/Inf."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -50.0, 50.0)

# ====== (Optional) per-band standardization ==================================
def fit_band_standardizer(X_train):
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_band_standardizer(X, mu, sigma):
    Z = (X - mu) / sigma
    return np.clip(Z, -20.0, 20.0)

# ====== Random subset helper ==================================================
def _random_subset_indices(n_total, n_keep=None, seed=REDUCE_SEED):
    if n_keep is None or n_keep > n_total:
        return np.arange(n_total)
    rs = np.random.RandomState(seed)
    return rs.choice(n_total, size=n_keep, replace=False)

# ====== I/O with optional sampling ===========================================
def readData(directory, max_files=None, random_state=REDUCE_SEED):
    """
    Load typical-only *.npy tiles from directory.
    If max_files is provided, randomly select that many files.
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
    If max_* provided, randomly select that many files from each.
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

def displayImages(X, imageSetIndx):
    numSets = 3
    grid_size = (numSets, 6)
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 3))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    i = 0
    for set_indx in imageSetIndx:
        img_array = X[set_indx, :, :, :]
        for j in range(grid_size[1]):
            axs[i, j].imshow(img_array[:, :, j])
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        i = i + 1

# ====== Plot signals (unchanged) =============================================
def plot_signals(X, num_sets):
    avgSignalsInSets = []
    for curSet in range(num_sets):
        image_set = X[curSet, :, :, :]
        n_images_in_set = image_set.shape[2]
        signal = np.zeros(image_set.shape[0] * image_set.shape[1])
        for i in range(n_images_in_set):
            signal = signal + image_set[:, :, i].ravel(order='F')
        signal = signal / n_images_in_set
        avgSignalsInSets.append(signal)
    grid_size = (5, 2)
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    index = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            axs[i, j].plot(avgSignalsInSets[index])
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            index = index + 1
            axs[i, j].set_title(f'Image set ({index})')

# ====== CP/Tucker decompose (robust) =========================================
def decompose_tensor_tucker(tensor, rank, *, mode="hosvd_fast"):
    """
    Fast Tucker per tile.
      mode="hosvd_fast": single-pass HOSVD (n_iter_max=1, init="svd")  ✅ fastest & stable
      mode="iter":       iterative ALS (n_iter_max=100)                ⚠️ slower
    Returns: (core, factors) as numpy arrays.
    """
    ranks = tuple(rank) if isinstance(rank, (list, tuple, np.ndarray)) else rank

    # --- Preferred: GPU/CPU current backend, float32 (GPU double is slow)
    try:
        Xb = _to_backend(tensor, use_float64=False)  # was True; use float32 for speed
        if mode == "hosvd_fast":
            core, factors = _tl_tucker(Xb, ranks, init="svd", n_iter_max=1, tol=0)  # positional ranks
        else:
            core, factors = _tl_tucker(Xb, ranks, init="svd", n_iter_max=100, tol=1e-5)
        core_np = _to_numpy(core)
        facs_np = [_to_numpy(Fm) for Fm in factors]
        return core_np.astype(np.float32), [Fm.astype(np.float32) for Fm in facs_np]
    except Exception:
        # Optional last resort: NumPy backend (explicitly opt-in to avoid thrash)
        old = tl.get_backend()
        try:
            tl.set_backend("numpy")
            if mode == "hosvd_fast":
                core, factors = _tl_tucker(tensor.astype(np.float32), ranks, init="svd", n_iter_max=1, tol=0)
            else:
                core, factors = _tl_tucker(tensor.astype(np.float32), ranks, init="svd", n_iter_max=100, tol=1e-5)
            return core.astype(np.float32), [Fm.astype(np.float32) for Fm in factors]
        finally:
            tl.set_backend(old)


def decompose_tensor_parafac(tensor, rank):
    try:
        Xb = _to_backend(tensor, use_float64=True)
        weights, factors = _tl_parafac(
            Xb, rank=rank, init="svd", n_iter_max=500, tol=1e-6,
            normalize_factors=True, random_state=42
        )
        facs_np = [ _to_numpy(Fm) for Fm in factors ]
        if all(np.all(np.isfinite(Fm)) for Fm in facs_np):
            return facs_np
    except Exception:
        pass
    try:
        old = tl.get_backend()
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            tensor.astype(np.float32), rank=rank, init="svd",
            n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=42
        )
        tl.set_backend(old)
        return [Fm.astype(np.float32) for Fm in factors]
    except Exception:
        raise RuntimeError("CP failed on all backends for this tile.")

# ====== Feature extractors ====================================================
def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

# ====== Tensor build/extract pipelines =======================================
def buildTensor(X, rank, num_sets, isTuckerDecomposition=True, ordered=False):
    """
    Decompose each tile.
    - CUDA backend: use a single worker to avoid GPU context thrash.
    - CPU backend: allow threads (still keep BLAS threads = 1 via env if possible).
    """
    use_cuda = (tl.get_backend() == "pytorch")
    try:
        import torch
        use_cuda = use_cuda and torch.cuda.is_available()
    except Exception:
        use_cuda = False

    # Pick worker count
    max_workers = None

    def decomp(i):
        if isTuckerDecomposition:
            return decompose_tensor_tucker(X[i], rank, mode="hosvd_fast")  # <- fast path
        else:
            return decompose_tensor_parafac(X[i], rank)

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

def extractFeatures(decomposed_data, num_sets, isTuckerDecomposition=True):
    if isTuckerDecomposition:
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(
                lambda i: extract_features_tucker(decomposed_data[i][0], decomposed_data[i][1]),
                range(num_sets)
            ))
    else:
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(
                lambda i: extract_features_cp(decomposed_data[i]),
                range(num_sets)
            ))
    return np.array(features)

# ====== Evaluation helpers ====================================================
def manual_auc(y_true, scores, positive_label=-1):
    """
    Manual ROC AUC: probability a random positive has a higher score than a random negative,
    counting ties as 0.5. Returns NaN if only one class present or scores are non-finite.
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

# ====== (Optional) CP/Tucker viz helper ======================================
def visualize_cp_scores(H, labels=None, title="CP sample coefficients (H)", use_pca=True):
    """
    Visualize CP sample coefficients (H).
    H: (n_samples, rank)
    labels: optional array-like for coloring points (e.g., +1 typical, -1 anomaly)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    H = np.asarray(H)
    n, r = H.shape

    # Heatmap (samples x components)
    H_std = StandardScaler(with_mean=True, with_std=True).fit_transform(H)
    plt.figure()
    plt.imshow(H_std, aspect='auto', interpolation='nearest')
    plt.xlabel("Component (r)")
    plt.ylabel("Sample")
    plt.title(f"{title} — heatmap (z-scored per component)")
    plt.colorbar()
    plt.tight_layout()

    # 2D view of samples in component space
    if r >= 2:
        if use_pca and r > 2:
            from sklearn.decomposition import PCA
            XY = PCA(n_components=2, random_state=0).fit_transform(H_std)
            subtitle = "PCA(2) of H"
        else:
            XY = H_std[:, :2]
            subtitle = "Components [0,1] of H"

        plt.figure()
        if labels is None:
            plt.scatter(XY[:, 0], XY[:, 1], s=12, alpha=0.8)
        else:
            # Map labels to colors/markers
            labels = np.asarray(labels)
            for val in np.unique(labels):
                m = labels == val
                plt.scatter(XY[m, 0], XY[m, 1], s=14, alpha=0.85, label=str(val))
            plt.legend(title="label")
        plt.title(f"{title} — {subtitle}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
    else:
        # rank == 1: show histogram of the lone coefficient
        plt.figure()
        if labels is None:
            plt.hist(H_std[:, 0], bins=30)
        else:
            for val in np.unique(labels):
                m = labels == val
                plt.hist(H_std[m, 0], bins=30, alpha=0.6, label=str(val))
            plt.legend(title="label")
        plt.title(f"{title} — component 0 distribution")
        plt.xlabel("z-scored coefficient")
        plt.ylabel("count")
        plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def cp_reconstruct_tile(A, B, C, h):
    """
    Reconstruct one tile from CP basis (A,B,C) and sample coefficients h.
    Shapes assumed: A:(I,R), B:(J,R), C:(K,R), h:(R,)
    Returns: Xhat:(I,J,K)
    """
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C); h = np.asarray(h).reshape(-1)
    # scale A’s columns by h_r, then sum over r of (a_r ⊗ b_r ⊗ c_r)
    Ah = A * h[np.newaxis, :]                        # (I,R)
    Xhat = np.einsum('ir,jr,kr->ijk', Ah, B, C, optimize=True)
    return Xhat

def visualize_cp_reconstruction(A, B, C, H, X_ref=None, idx=0, bands=(0,1,2,3,4,5), title_prefix=""):
    """
    Reconstruct tile `idx` from H and optionally show original beside it.
    - A,B,C: CP basis
    - H: (n_samples, R)
    - X_ref: optional original data array (n_samples, I, J, K) for side-by-side
    - bands: which spectral bands to display
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

# ====== Global CP basis + projection for OC-SVM ==============================
def fit_global_cp_basis(X_train, rank, random_state=42, max_train_samples=None, use_gpu=USE_GPU_CP):
    """
    Fit one global CP model to TRAIN tensor (N,64,64,6).
    Returns:
      (A,B,C) as np.float32 and H_train as np.float32 (N,R).
    If subsampling is used, H_train is recomputed for the full TRAIN via projection.
    """
    import numpy as _np
    N = X_train.shape[0]
    X_in = X_train
    if max_train_samples is not None and N > max_train_samples:
        idx = np.random.RandomState(123).choice(N, size=max_train_samples, replace=False)
        X_in = X_in[idx]

    # GPU branch (TensorLy+PyTorch)
    if use_gpu and tl.get_backend() == "pytorch":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.manual_seed(random_state)
            with torch.no_grad():
                Xb = _to_backend(X_in, use_float64=False)   # float32 on GPU is fine
                weights, factors = _tl_parafac(
                    Xb, rank=rank, init="random",
                    n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
                )
                H_t, A_t, B_t, C_t = factors   # H (N'×R), A (64×R), B (64×R), C (6×R)
                H_t = H_t * weights[None, :]

                # Convert basis to NumPy
                A = A_t.detach().cpu().numpy().astype(_np.float32)
                B = B_t.detach().cpu().numpy().astype(_np.float32)
                C = C_t.detach().cpu().numpy().astype(_np.float32)

                if X_in.shape[0] != X_train.shape[0]:
                    # Recompute H for the FULL training set via projection
                    H_full = project_cp_coeffs_torch(X_train, A, B, C, device="cuda")
                    return (A, B, C), H_full.astype(_np.float32)
                else:
                    H_np = H_t.detach().cpu().numpy().astype(_np.float32)
                    return (A.astype(_np.float32), B.astype(_np.float32), C.astype(_np.float32)), H_np
        # fall through to CPU if no CUDA

    # CPU fallback (NumPy)
    old_backend = tl.get_backend()
    try:
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            X_in.astype(_np.float32), rank=rank, init="random",
            n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
        )
        H, A, B, C = [ _np.asarray(F) for F in factors ]
        lam = _np.asarray(weights)
        H = H * lam[None, :]
        if X_in.shape[0] != X_train.shape[0]:
            A = A.astype(_np.float32); B = B.astype(_np.float32); C = C.astype(_np.float32)
            Ginv = precompute_cp_projection(A, B, C)
            H_full = project_cp_coeffs(X_train, A, B, C, Ginv=Ginv)
            return (A, B, C), H_full.astype(_np.float32)
        else:
            return (A.astype(_np.float32), B.astype(_np.float32), C.astype(_np.float32)), H.astype(_np.float32)
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

# ====== Torch-based projection (used in GPU path) ============================
def project_cp_coeffs_torch(X, A, B, C, device="cuda", batch=256, eps=1e-8):
    """
    Torch (GPU/CPU) projection of a batch X: (N,64,64,6) onto fixed CP basis (A,B,C).
    Returns H: (N,R) as np.float32.
    """
    import torch
    assert X.ndim == 4 and X.shape[1:] == (64, 64, 6), "X must be (N,64,64,6)"
    A_t = torch.as_tensor(A, dtype=torch.float32, device=device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    C_t = torch.as_tensor(C, dtype=torch.float32, device=device)

    # Precompute Ginv = ((A^T A) * (B^T B) * (C^T C))^{-1}
    AtA = A_t.T @ A_t
    BtB = B_t.T @ B_t
    CtC = C_t.T @ C_t
    G = AtA * BtB * CtC
    R = G.shape[0]
    G = G + eps * torch.eye(R, device=device, dtype=G.dtype)
    Ginv = torch.linalg.inv(G)

    N = X.shape[0]
    H_out = torch.empty((N, R), dtype=torch.float32, device=device)

    for s in range(0, N, batch):
        e = min(s + batch, N)
        Xb = torch.as_tensor(X[s:e], dtype=torch.float32, device=device)  # (B,64,64,6)

        # g[n,r] = <X_n, a[:,r] ∘ b[:,r] ∘ c[:,r]>
        t1 = torch.einsum('nijk,ir->nrjk', Xb, A_t)   # (B,R,64,6)
        t2 = torch.einsum('nrjk,jr->nrk', t1, B_t)    # (B,R,6)
        g  = torch.einsum('nrk,kr->nr', t2, C_t)      # (B,R)

        H_out[s:e] = (g @ Ginv).to(torch.float32)

        del Xb, t1, t2, g
        if isinstance(device, str) and device.startswith('cuda'):
            torch.cuda.empty_cache()

    return H_out.detach().cpu().numpy().astype(np.float32)

def ocsvm_gamma_grid_for_dim(d):
    base = 1.0 / max(d, 1)
    return [base * t for t in (0.1, 0.3, 1.0, 3.0, 10.0)]

# ====== NEW: Auto-select best single CP component ============================
def select_best_single_cp_component(Htr_s, Hval_s, y_val):
    """
    Choose the best single column k of H (after scaling) using VAL:
      - If y_val has both classes, maximize VAL AUC (anomaly-positive).
      - If y_val is None or single-class, minimize false-positive rate on VAL
        at the One-Class SVM default threshold.
    Returns: k_best (int)
    """
    has_labels = (y_val is not None) and np.isin(-1, y_val).any() and np.isin(1, y_val).any()

    best_score = -np.inf
    k_best = 0

    # Tiny grid is sufficient in 1-D
    nus = [0.02, 0.05, 0.1, 0.2]
    gammas = [1.0, 0.3, 0.1, "scale"]

    for k in range(Htr_s.shape[1]):
        Xtr = Htr_s[:, [k]]
        Xva = Hval_s[:, [k]]
        for nu in nus:
            for gamma in gammas:
                try:
                    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(Xtr)
                    s_val = -model.decision_function(Xva).ravel()
                    if not np.all(np.isfinite(s_val)):
                        continue
                    if has_labels:
                        auc = manual_auc(y_val, s_val, positive_label=-1)
                        if np.isfinite(auc) and auc > best_score:
                            best_score = auc
                            k_best = k
                    else:
                        fp = float((s_val >= 0.0).mean())  # default cutoff: 0 in decision_function
                        score = -fp  # lower FP is better
                        if score > best_score:
                            best_score = score
                            k_best = k
                except Exception:
                    continue

    return int(k_best)

# ====== NEW: Read everything once & freeze the split =========================
def prepare_data_once(val_fraction=VAL_FRACTION, random_state=42):
    """
    Loads TRAIN / VAL / TEST (with optional reduction) exactly once,
    and returns a dict bundle consumed by parafac_OC_SVM and rank searches.

    Returns dict with keys:
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
        print(f"[VAL] Using separate validation dir (typical-only): {validation_data} (N={X_val.shape[0]})")
    else:
        print("[VAL] Separate validation dir missing/empty; using stratified VAL/FINAL split from test pool.")
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
        "y_val": y_val,   # may be None if separate typical-only VAL dir exists
        "y_fin": y_fin
    }

# ====== NEW: Common split fetch + optional band standardization ==============
def get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE):
    """
    Returns (X_train, X_val, X_fin, y_val, y_fin, mu_b, sig_b) where X_* are optionally
    standardized using TRAIN stats. mu_b/sig_b will be None if standardize=False.
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
        print("[sanity] Cleaning TRAIN tiles ..."); X_train = clean_tiles(X_train)
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("[sanity] Cleaning VAL tiles ...");   X_val = clean_tiles(X_val)
    if np.isnan(X_fin).any() or np.isinf(X_fin).any():
        print("[sanity] Cleaning FINAL tiles ..."); X_fin = clean_tiles(X_fin)

    return X_train, X_val, X_fin, y_val, y_fin, mu_b, sig_b

# ====== NEW: Single place for CP fit + project to H ==========================
def cp_fit_and_project(X_train, X_val, X_fin, rank, random_state=42, cp_basis_max_train_samples=None):
    """
    Fit global CP on X_train and project X_val/X_fin into coefficients.
    Returns: (A,B,C), H_train, H_val, H_fin
    """
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state,
        max_train_samples=cp_basis_max_train_samples
    )
    Ginv = precompute_cp_projection(A, B, C)
    H_val = project_cp_coeffs(X_val, A, B, C, Ginv=Ginv)
    H_fin = project_cp_coeffs(X_fin, A, B, C, Ginv=Ginv)
    return (A, B, C), H_train, H_val, H_fin

# ====== CP + OC-SVM using preloaded data =====================================
def parafac_OC_SVM(rank, data_bundle,
                   displayConfusionMatrix=False, use_pca_whiten=True,
                   random_state=42, cp_basis_max_train_samples=None):
    """
    Pipeline (no I/O):
      1) Standardize using TRAIN stats (optional).
      2) Fit a single global CP basis on TRAIN (optionally subsampled).
      3) Project VAL and FINAL onto basis.
      4) Select OC-SVM params on VAL (AUC if labels present, else one-class).
      5) Evaluate on FINAL.
    """
    start_time = time.time()

    # ---- Common split & standardization
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

    # ---- GLOBAL CP fit + project (COMMON)
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state,
        cp_basis_max_train_samples=cp_basis_max_train_samples
    )

    if RUN_VISUALIZATION:
        visualize_cp_scores(H_train, labels=None, title=f"H_train (rank={rank})")
        visualize_cp_reconstruction(A, B, C, H_train, X_ref=X_train, idx=0, bands=(0, 1, 2))

    # ---- Scale
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    # ---- Choose features: single component (auto) or full H (with/without PCA)
    if use_pca_whiten:
        pca   = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Htr_w  = pca.fit_transform(Htr_s)
        Hval_w = pca.transform(Hval_s)
        Hfin_w = pca.transform(Hfin_s)
        feat_dim = Htr_w.shape[1]
    else:
        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s
        feat_dim = Htr_w.shape[1]

    # ---- OC-SVM hyperparams (γ scaled to 1/d)
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(feat_dim),
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    # Selection mode: AUC if VAL has labels (both classes), else one-class criterion
    use_auc_on_val = (y_val is not None) and (np.isin(-1, y_val).any() and np.isin(1, y_val).any())

    best_score_tuple = None
    best_model = None
    best_params = None
    best_aux_print = ""

    for params in ParameterGrid(param_grid):
        try:
            model = OneClassSVM(**params).fit(Htr_w)
            s_val = -model.decision_function(Hval_w).ravel()  # larger => more anomalous
            if not np.all(np.isfinite(s_val)):
                continue

            if use_auc_on_val:
                auc = manual_auc(y_val, s_val, positive_label=-1)
                if not np.isfinite(auc):
                    continue
                score_tuple = (-auc,)  # minimize negative AUC == maximize AUC
                aux = f"AUC={auc:.3f}"
            else:
                # One-class selection (VAL typical-only): minimize FP@0, then P95, then mean
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

        except Exception:
            continue

    if best_model is None:
        print("[CP+OCSVM] No valid model on validation; falling back to simple RBF defaults.")
        for params in [{"kernel": "rbf", "gamma": 1.0 / max(Htr_w.shape[1], 1), "nu": 0.1},
                       {"kernel": "rbf", "gamma": "scale", "nu": 0.1}]:
            try:
                model = OneClassSVM(**params).fit(Htr_w)
                best_model = model; best_params = params
                best_aux_print = "(fallback)"
                break
            except Exception:
                continue
        if best_model is None:
            # Degenerate final fallback: distance-to-mean scorer
            print("[CP+OCSVM] Fallback failed; using distance-to-mean scorer.")
            mu = Htr_w.mean(axis=0, keepdims=True)
            s_fin = np.linalg.norm(Hfin_w - mu, axis=1)
            auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
            print(f"[Fallback] FINAL AUC={auc_fin:.3f} | Acc={acc_opt:.3f}")
            print('runtime:', round(time.time() - start_time, 2), 's')
            return acc_opt

    sel_mode = "AUC" if use_auc_on_val else "one-class"
    print(f"[CP+OCSVM] VAL ({sel_mode}) picked params={best_params} with {best_aux_print}")

    # ---- FINAL evaluation
    s_fin = -best_model.decision_function(Hfin_w).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    print(f"[CP+OCSVM] FINAL AUC={auc_fin:.3f}")

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"[CP+OCSVM] Chosen threshold={th_opt:.6f} | Accuracy={acc_opt:.3f}")
    if displayConfusionMatrix:
        y_pred_thresh = np.where(s_fin >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot(); plt.show()

    print('runtime:', round(time.time() - start_time, 2), 's')
    return acc_opt, auc_fin

# ====== Optional raw-pixel OC-SVM (unchanged) ================================
def ocsvm_raw_geography(displayConfusionMatrix=False):
    # === Speed toggles (safe defaults) =======================================
    _USE_PCA = False
    _PCA_COMPONENTS = 256
    _PARALLEL_GRID = True
    _SVM_CACHE_MB = 1024
    _SVM_TOL = 1e-2
    _SVM_MAX_ITER = -1
    _USE_NYSTROEM = False
    _NY_COMPONENTS = 512

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import ParameterGrid
    from sklearn import metrics
    import matplotlib.pyplot as plt

    if _USE_PCA:
        from sklearn.decomposition import PCA
    if _USE_NYSTROEM:
        from sklearn.kernel_approximation import Nystroem
    if _PARALLEL_GRID:
        from joblib import Parallel, delayed

    # === Load data ===========================================================
    X_train, _ = readData(
        train_data,
        max_files=(REDUCE_TRAIN_N if REDUCE_DATASETS else None),
        random_state=REDUCE_SEED
    )
    X_test, true_labels = readData_test(
        test_typical_data, test_anomaly_data,
        max_typical=(REDUCE_TEST_TYP_N if REDUCE_DATASETS else None),
        max_anomaly=(REDUCE_TEST_ANO_N if REDUCE_DATASETS else None),
        random_state=REDUCE_SEED
    )

    # === Per-band standardization (if enabled) ===============================
    if USE_BAND_STANDARDIZE:
        mu, sig = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu, sig)
        X_test  = apply_band_standardizer(X_test,  mu, sig)

    # === Flatten + scale (avoid copies) ======================================
    n_train = X_train.shape[0]; n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat  = X_test.reshape(n_test,  -1)

    scaler = StandardScaler(copy=False)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    if _USE_PCA:
        pca = PCA(n_components=_PCA_COMPONENTS, svd_solver='randomized', random_state=REDUCE_SEED)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled  = pca.transform(X_test_scaled)

    def fit_eval_linear_on_map(gamma):
        mapper = Nystroem(gamma=gamma, n_components=_NY_COMPONENTS, random_state=REDUCE_SEED)
        Z_train = mapper.fit_transform(X_train_scaled)
        Z_test  = mapper.transform(X_test_scaled)
        return Z_train, Z_test

    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    best_acc = -1; best_model = None; best_params = None
    y_true = np.asarray(true_labels)

    def _train_eval(params):
        p = dict(params)
        p.setdefault('cache_size', _SVM_CACHE_MB)
        p.setdefault('tol', _SVM_TOL)
        p.setdefault('max_iter', _SVM_MAX_ITER)

        if _USE_NYSTROEM and p.get('kernel', 'rbf') == 'rbf' and p['gamma'] not in ('scale', 'auto'):
            Z_train, Z_test = fit_eval_linear_on_map(p['gamma'])
            p_lin = dict(nu=p['nu'], kernel='linear', cache_size=_SVM_CACHE_MB, tol=_SVM_TOL, max_iter=_SVM_MAX_ITER)
            model = OneClassSVM(**p_lin).fit(Z_train)
            preds = model.predict(Z_test)
        else:
            model = OneClassSVM(**p).fit(X_train_scaled)
            preds = model.predict(X_test_scaled)

        acc = float(np.mean(preds == y_true))
        return acc, params, model

    grid_list = list(ParameterGrid(param_grid))
    if _USE_NYSTROEM:
        grid_list = [g for g in grid_list if (g['kernel'] == 'rbf' or not _USE_NYSTROEM)]

    if _PARALLEL_GRID:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, prefer="threads", verbose=0)(
            delayed(_train_eval)(p) for p in grid_list
        )
    else:
        results = [_train_eval(p) for p in grid_list]

    for acc, params, model in results:
        if acc > best_acc:
            best_acc, best_params, best_model = acc, params, model

    print(f"Best accuracy (grid @ default threshold 0): {best_acc:.3f} with params: {best_params}")

    if _USE_NYSTROEM and best_params.get('kernel', 'rbf') == 'rbf' and best_params['gamma'] not in ('scale', 'auto'):
        Z_train, Z_test = fit_eval_linear_on_map(best_params['gamma'])
        predictions_default = best_model.predict(Z_test)
        scores_normal = best_model.decision_function(Z_test).ravel()
    else:
        predictions_default = best_model.predict(X_test_scaled)
        scores_normal = best_model.decision_function(X_test_scaled).ravel()

    scores_anom = -scores_normal  # anomaly-positive
    accuracy_default = float(np.mean(predictions_default == y_true))
    print("Accuracy @ default cutoff (0):", accuracy_default)
    auc_manual = manual_auc(y_true, scores_anom, positive_label=-1)
    print("ROC AUC (manual, anomaly-positive):", auc_manual)

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_true, scores_anom, positive_label=-1)
    print(f"Chosen threshold (max-accuracy on ROC scores): {th_opt:.6f}  |  Accuracy @ chosen: {acc_opt:.3f}")

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(y_true, np.where(scores_anom >= th_opt, -1, 1), labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot(); plt.show()

    return acc_opt, best_params


def one_class_svm():
    print('One Class SVM')
    accuracy, param = ocsvm_raw_geography(False)
    print('One class SVM best accuracy:', accuracy, 'param:', param)

# ====== Rank search now reuses the preloaded bundle ==========================
def cp_rank_search_one_class_svm(data_bundle):
    print('CP rank search One Class SVM')
    startRank = 10; endRank = 385; step = 5  # tighter range for speed
    rank_score = {}
    for rank in range(startRank, endRank, step):
        print('Rank:', rank)
        acc, auc = parafac_OC_SVM(rank, data_bundle, use_pca_whiten=True)
        rank_score[rank] = auc
        print('Accuracy:', acc, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]

# ====== NEW: PCA + OC-SVM toggle =============================================
enable_pca_oc_svm = False   # set True to run the PCA-on-raw-pixels OC-SVM path
PCA_N_COMPONENTS = 32       # typical sweet spot: 128–512 (<= min(N-1, D))
PCA_WHITEN = False          # PCA whitening helps OC-SVM RBF stability
PCA_RANDOMIZED = False      # randomized SVD for speed on high-D

# ====== NEW: PCA + OC-SVM on raw pixels ======================================
def pca_OC_SVM(data_bundle,
               n_components=PCA_N_COMPONENTS,
               whiten=PCA_WHITEN,
               randomized=PCA_RANDOMIZED,
               displayConfusionMatrix=False,
               random_state=42):
    """
    Raw-pixels pipeline:
      1) Optional per-band standardization (fit on TRAIN).
      2) Flatten -> StandardScaler (fit on TRAIN).
      3) PCA on TRAIN only, then transform VAL & FINAL.
      4) Select OC-SVM params on VAL:
         - If VAL has both classes: maximize AUC (anomaly positive).
         - If typical-only: minimize FP@0 (tie-break: P95, mean).
      5) Evaluate on FINAL, print AUC & max-accuracy threshold.
    """
    start_time = time.time()

    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

    # ---- Flatten -> scale
    Ntr = X_train.shape[0]
    Xtr_flat  = X_train.reshape(Ntr, -1)
    Xval_flat = X_val.reshape(X_val.shape[0], -1)
    Xfin_flat = X_fin.reshape(X_fin.shape[0], -1)

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr_flat)
    Xval_s = scaler.transform(Xval_flat)
    Xfin_s = scaler.transform(Xfin_flat)

    # ---- PCA on TRAIN only
    D = Xtr_s.shape[1]
    max_allowed = max(1, min(D, Ntr - 1))
    n_comp = int(max(1, min(n_components, max_allowed)))
    svd_solver = 'randomized' if randomized else 'auto'
    pca = PCA(n_components=n_comp, whiten=whiten, svd_solver=svd_solver, random_state=random_state)

    Ztr  = pca.fit_transform(Xtr_s)
    Zval = pca.transform(Xval_s)
    Zfin = pca.transform(Xfin_s)

    feat_dim = Ztr.shape[1]
    print(f"[PCA+OCSVM] PCA dims: D={D} -> d={feat_dim} (whiten={whiten}, solver={svd_solver})")

    # ---- Hyperparameter search (γ scaled by 1/d)
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(feat_dim),
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    use_auc_on_val = (y_val is not None) and (np.isin(-1, y_val).any() and np.isin(1, y_val).any())

    best_score_tuple = None
    best_model = None
    best_params = None
    best_aux_print = ""

    for params in ParameterGrid(param_grid):
        try:
            model = OneClassSVM(**params).fit(Ztr)
            s_val = -model.decision_function(Zval).ravel()
            if not np.all(np.isfinite(s_val)):
                continue

            if use_auc_on_val:
                auc = manual_auc(y_val, s_val, positive_label=-1)
                if not np.isfinite(auc):
                    continue
                score_tuple = (-auc,)  # maximize AUC
                aux = f"AUC={auc:.3f}"
            else:
                fp_rate = float((s_val >= 0.0).mean())  # default cutoff 0
                p95 = float(np.percentile(s_val, 95))
                mean_s = float(np.mean(s_val))
                score_tuple = (fp_rate, p95, mean_s)
                aux = f"FP={fp_rate:.3f}, P95={p95:.4f}, mean={mean_s:.4f}"

            if (best_score_tuple is None) or (score_tuple < best_score_tuple):
                best_score_tuple = score_tuple
                best_model = model
                best_params = params
                best_aux_print = aux

        except Exception:
            continue

    if best_model is None:
        print("[PCA+OCSVM] No valid model on validation; using simple defaults.")
        defaults = [{"kernel": "rbf", "gamma": 1.0 / max(feat_dim, 1), "nu": 0.1},
                    {"kernel": "rbf", "gamma": "scale", "nu": 0.1}]
        for params in defaults:
            try:
                best_model = OneClassSVM(**params).fit(Ztr)
                best_params = params
                best_aux_print = "(fallback)"
                break
            except Exception:
                pass
        if best_model is None:
            # last-resort distance-to-mean on Z
            mu = Ztr.mean(axis=0, keepdims=True)
            s_fin = np.linalg.norm(Zfin - mu, axis=1)
            auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
            print(f"[Fallback PCA] FINAL AUC={auc_fin:.3f} | Acc={acc_opt:.3f}")
            print('runtime:', round(time.time() - start_time, 2), 's')
            return acc_opt

    sel_mode = "AUC" if use_auc_on_val else "one-class"
    print(f"[PCA+OCSVM] VAL ({sel_mode}) picked params={best_params} with {best_aux_print}")

    # ---- FINAL evaluation
    s_fin = -best_model.decision_function(Zfin).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    print(f"[PCA+OCSVM] FINAL AUC={auc_fin:.3f}")

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"[PCA+OCSVM] Chosen threshold={th_opt:.6f} | Accuracy={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(s_fin >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot(); plt.show()

    print('runtime:', round(time.time() - start_time, 2), 's')
    return acc_opt

# =============================================================================
# === NEW: Tucker with OC-SVM (now uses common data via data_bundle) ==========
# =============================================================================
def tucker_one_class_svm(rank, data_bundle, displayConfusionMatrix=False, random_state=42, val_fraction=0.5):
    """
    Tucker + OC-SVM now shares the same read/standardize path via data_bundle.
    """
    # ---- Common split & standardization
    X_train, X_val, X_fin, y_val, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)

    # ---- Tucker decompositions
    n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
    start_time = time.time()
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_va = buildTensor(X_val,   rank, n_va, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)
    print(f"Decomposition time: {time.time() - start_time:.2f} seconds")

    # ---- Feature extraction + scaling (fit on TRAIN only)
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True)
    Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_va = scaler.transform(Feat_va)
    Z_fi = scaler.transform(Feat_fi)

    # ---- Hyperparameter search on VAL
    d = Z_tr.shape[1]
    gamma_grid = [1.0 / max(d, 1) * t for t in (0.1, 0.3, 1.0, 3.0, 10.0)] + ["scale"]
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  gamma_grid,
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2],
    }

    use_auc_on_val = (y_val is not None) and (np.isin(-1, y_val).any() and np.isin(1, y_val).any())

    best_tuple = None      # objective tuple to minimize
    best_model = None
    best_params = None
    best_aux = ""

    for p in ParameterGrid(param_grid):
        try:
            m = OneClassSVM(**p).fit(Z_tr)
            s_val = -m.decision_function(Z_va).ravel()  # anomaly-positive scores
            if not np.all(np.isfinite(s_val)):
                continue

            if use_auc_on_val:
                auc = manual_auc(y_val, s_val, positive_label=-1)
                if not np.isfinite(auc):
                    continue
                obj = (-auc,)  # maximize AUC
                aux = f"AUC={auc:.3f}"
            else:
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
        except Exception:
            continue

    if best_model is None:
        # reasonable defaults if VAL selection fails
        for p in [{"kernel": "rbf", "gamma": 1.0 / max(d, 1), "nu": 0.1},
                  {"kernel": "rbf", "gamma": "scale",           "nu": 0.1}]:
            try:
                best_model = OneClassSVM(**p).fit(Z_tr)
                best_params = p
                best_aux = "(fallback)"
                break
            except Exception:
                pass
        if best_model is None:
            # last-resort: distance-to-mean scorer on FINAL
            mu = Z_tr.mean(axis=0, keepdims=True)
            s_fin = np.linalg.norm(Z_fi - mu, axis=1)
            auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
            th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
            print(f"[Tucker+OCSVM Fallback] FINAL AUC={auc_fin:.3f} | Acc={acc_opt:.3f}")
            return acc_opt

    sel_mode = "AUC" if use_auc_on_val else "one-class"
    print(f"[Tucker+OCSVM] VAL ({sel_mode}) picked params={best_params} with {best_aux}")

    # ---- FINAL evaluation
    s_fin = -best_model.decision_function(Z_fi).ravel()
    auc_fin = manual_auc(y_fin, s_fin, positive_label=-1)
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
    print(f"[Tucker+OCSVM] FINAL AUC={auc_fin:.3f} | Acc@best={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred = np.where(s_fin >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"]).plot()
        plt.show()

    return acc_opt, auc_fin


def tucker_rank_search_one_class_svm(data_bundle):
    """
    Grid over Tucker rank triples; each trial uses the validation-aware
    `tucker_one_class_svm` above. Chooses the rank with highest FINAL accuracy.
    """
    print('Tucker rank search One Class SVM')
    rankSet = sorted({5, 16, 32, 64})
    rank_score = {}
    for i in rankSet:
        for j in rankSet:
            for k in sorted({5, 16}):
                r = (i, j, k)
                print('Rank:', i, j, k)
                acc, auc = tucker_one_class_svm(r, data_bundle)
                rank_score[r] = auc
                print('Accuracy:', acc, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]

# ===============================================
# CP with Autoencoder (now uses common CP fit/proj + common data)
# ===============================================
def parafac_autoencoder(rank, factor, bottleneck, data_bundle,
                        displayConfusionMatrix=False,
                        random_state=42, cp_basis_max_train_samples=None, use_pca_whiten=False):
    """
    CP+Autoencoder using a *single global CP basis* fit on TRAIN, then projecting
    VAL/FINAL onto that basis (same pattern as parafac_OC_SVM).

    Prints FINAL AUC and max-accuracy threshold metrics (parity with CP+OCSVM).
    Returns accuracy using the VAL/TRAIN-derived threshold.
    """
    # ---- Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # ---- GLOBAL CP -> H matrices (COMMON)
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state,
        cp_basis_max_train_samples=cp_basis_max_train_samples
    )

    # ---- Scale features (and optional PCA whitening) fit on TRAIN only
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

    # ---- Define Autoencoder (simple MLP bottleneck)
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

    # ---- Threshold from VAL
    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean(np.square(Z_va - recon_va), axis=1)
    threshold = np.percentile(err_va, 95)

    # ---- Score FINAL
    recon_fi = autoencoder.predict(Z_fi, verbose=0)
    err_fi = np.mean(np.square(Z_fi - recon_fi), axis=1)  # anomaly-positive scores

    preds_default = (err_fi > threshold).astype(int)
    preds_default[preds_default == 1] = -1
    preds_default[preds_default == 0] = 1
    accuracy_default = float(np.mean(preds_default == y_fin))
    print(f"[CP+AE(global)] Accuracy @ VAL-derived threshold: {accuracy_default:.3f}")

    auc_fin = manual_auc(y_fin, err_fi, positive_label=-1)
    print(f"[CP+AE(global)] FINAL AUC={auc_fin:.3f}")

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_fi, positive_label=-1)
    print(f"[CP+AE(global)] Chosen threshold (max-accuracy on recon error): {th_opt:.6f} | Accuracy={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(err_fi >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal']).plot()
        plt.show()

    return acc_opt, auc_fin


def autoencoder_anomaly(data_bundle, factor, bottleneck, displayConfusionMatrix=False):
    """
    Raw-pixel autoencoder that now uses the shared data bundle + standardization.
    """
    # ---- Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Flatten + scale (fit on TRAIN)
    n_tr = X_train.shape[0]
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_train.reshape(n_tr, -1))

    n_va = X_val.shape[0]
    Z_va = scaler.transform(X_val.reshape(n_va, -1))

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

    # TEST/FINAL
    n_te = X_fin.shape[0]
    Z_te = scaler.transform(X_fin.reshape(n_te, -1))

    # Threshold from VAL (typical-only)
    recon_va = autoencoder.predict(Z_va, verbose=0)
    thr = np.percentile(np.mean(np.square(Z_va - recon_va), axis=1), 95)

    # Final scoring
    recon_te = autoencoder.predict(Z_te, verbose=0)
    err_te = np.mean(np.square(Z_te - recon_te), axis=1)  # anomaly-positive scores

    preds_default = (err_te > thr).astype(int)
    preds_default[preds_default == 1] = -1
    preds_default[preds_default == 0] = 1
    accuracy_default = float(np.mean(preds_default == y_fin))
    print(f"[AE(raw)] Accuracy @ VAL-derived threshold: {accuracy_default:.3f}")

    auc_fin = manual_auc(y_fin, err_te, positive_label=-1)
    print(f"[AE(raw)] FINAL AUC={auc_fin:.3f}")

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_te, positive_label=-1)
    print(f"[AE(raw)] Chosen threshold (max-accuracy on recon error): {th_opt:.6f} | Accuracy={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(err_te >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal']).plot()
        plt.show()

    return acc_opt, auc_fin


def autoencoder(data_bundle, displayConfusionMatrix=False, sweep_factors=(1, 2, 3), sweep_bottlenecks=(16, 32, 64)):
    """
    Hyperparam sweep for the raw-pixel autoencoder (no CP/Tucker).
    Uses `autoencoder_anomaly(...)`, which honors the validation dataset
    (typical-only) for both early stopping and threshold calibration.
    """
    print('Autoencoder (raw-pixel) — sweeping factors and bottlenecks')
    param_accuracy = {}
    param_auc = {}
    best_acc = -1.0
    best_params = None

    for factor in sweep_factors:
        for bottleneck in sweep_bottlenecks:
            print('Factor:', factor, 'Bottleneck:', bottleneck)
            acc, auc = autoencoder_anomaly(data_bundle, factor, bottleneck, displayConfusionMatrix=False)
            param_accuracy[(factor, bottleneck)] = acc
            param_auc[(factor, bottleneck)] = auc
            print('Accuracy', acc)
            if acc > best_acc:
                best_acc = acc
                best_params = (factor, bottleneck)

    print('Param accuracy', param_accuracy)
    print('Best param for autoencoder', best_params, best_acc)

    if displayConfusionMatrix and best_params is not None:
        autoencoder_anomaly(data_bundle, best_params[0], best_params[1], displayConfusionMatrix=True)

    return best_params, best_acc

def cp_rank_search_autoencoder(data_bundle):
    print('CP rank search autoencoder')
    startRank = 10
    endRank = 385
    step = 5
    rank_score = {}
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            for i in range(startRank, endRank, step):
                rank = i
                print('Factor:', factor, 'Bottleneck:', bottleneck, 'Rank:', i)
                accuracy, auc = parafac_autoencoder(rank, factor, bottleneck, data_bundle)
                rank_score[(rank, factor, bottleneck)] = auc
                print('Accuracy', accuracy, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]

# =============================================================================
# === NEW STRATEGY: Tucker with Autoencoder (converted with minimal changes) ==
# =============================================================================
def tucker_neural_network_autoencoder(rank, factor, bottleneck, data_bundle,
                                      displayConfusionMatrix=False):
    """
    Converted from your snippet to use the common data/validation path and
    to print AUC + max-accuracy threshold (parity with other strategies).
    """
    # ---- Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # ---- Tucker decomposition per split
    n_tr, n_va, n_fi = X_train.shape[0], X_val.shape[0], X_fin.shape[0]
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_va = buildTensor(X_val,   rank, n_va, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)

    # ---- Extract and normalize features (fit scaler on TRAIN only)
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True)
    Feat_va = extractFeatures(decomp_va, n_va, isTuckerDecomposition=True)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_va = scaler.transform(Feat_va)
    Z_fi = scaler.transform(Feat_fi)

    # ---- Define the autoencoder model
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

    # Early stopping callback (use separate VAL split)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(Z_tr, Z_tr, epochs=10, batch_size=32,
                    validation_data=(Z_va, Z_va),
                    callbacks=[early_stopping], verbose=0)

    # ---- Threshold from VAL (typical-only directory, if provided)
    recon_va = autoencoder.predict(Z_va, verbose=0)
    err_va = np.mean(np.square(Z_va - recon_va), axis=1)
    threshold = np.percentile(err_va, 95)  # assume anomalies rare / typical-heavy VAL

    # ---- Predict on FINAL
    recon_fi = autoencoder.predict(Z_fi, verbose=0)
    err_fi = np.mean(np.square(Z_fi - recon_fi), axis=1)  # anomaly-positive scores

    # Default threshold accuracy
    predictions = (err_fi > threshold).astype(int)
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1
    accuracy_default = float(np.mean(predictions == y_fin))
    print(f"[Tucker+AE] Accuracy @ VAL-derived threshold: {accuracy_default:.3f}")

    # AUC + max-accuracy threshold (for parity with other strategies)
    auc_fin = manual_auc(y_fin, err_fi, positive_label=-1)
    print(f"[Tucker+AE] FINAL AUC={auc_fin:.3f}")

    th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, err_fi, positive_label=-1)
    print(f"[Tucker+AE] Chosen threshold (max-accuracy on recon error): {th_opt:.6f} | Accuracy={acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(err_fi >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_fin, y_pred_thresh, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal']).plot()
        plt.show()

    return acc_opt, auc_fin


def tucker_rank_search_autoencoder(data_bundle):
    """
    Minimal-conversion sweep using your original loop structure.
    """
    print('Tucker rank search autoencoder')
    rankSet = sorted({5, 16, 32, 64})
    rank_score = {}
    # for factor in range(1, 4):
    for factor in range(3, 4):  # keep your original choice
        for bottleneck in {16, 32, 64}:
            for i in rankSet:
                for j in rankSet:
                    for k in sorted({5, 16}):
                        rank = (i, j, k)
                        print('Rank:', i, j, k, 'Factor', factor, 'Bottleneck:', bottleneck)
                        accuracy, auc = tucker_neural_network_autoencoder(rank, factor, bottleneck, data_bundle)
                        rank_score[(rank, factor, bottleneck)] = auc
                        print('Accuracy:', accuracy, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]

def _if_mean_score_scorer(estimator, X, y=None):
    # Higher is better (less "anomalous" on average)
    try:
        return float(np.mean(estimator.score_samples(X)))
    except Exception:
        return -np.inf

# =============================================================================
# === CP (global basis) + Isolation Forest ====================================
# =============================================================================
def parafac_isolation_forest(rank, data_bundle,
                          displayConfusionMatrix=False,
                          random_state=42,
                          cp_basis_max_train_samples=None,
                          use_pca_whiten=False):
    """
    CP features via the shared global CP basis -> project to H, then IsolationForest.
    Evaluates on FINAL (test pool) with labels y_fin; prints AUC and returns accuracy.
    """
    # Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Global CP fit + project to coefficients (H)
    (A, B, C), H_train, H_val, H_fin = cp_fit_and_project(
        X_train, X_val, X_fin, rank,
        random_state=random_state,
        cp_basis_max_train_samples=cp_basis_max_train_samples
    )

    # Scale (fit on TRAIN only)
    scaler = StandardScaler()
    Htr_s = scaler.fit_transform(H_train)
    Hva_s = scaler.transform(H_val)
    Hfi_s = scaler.transform(H_fin)

    # Optional PCA whitening on H (kept off by default for minimal changes)
    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Z_tr = pca.fit_transform(Htr_s)
        Z_va = pca.transform(Hva_s)
        Z_fi = pca.transform(Hfi_s)
    else:
        Z_tr, Z_va, Z_fi = Htr_s, Hva_s, Hfi_s

    # Hyperparameter tuning (unsupervised) on TRAIN only
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
    iso = IsolationForest()
    grid = GridSearchCV(
        estimator=iso,
        param_grid=param_grid,
        cv=3,
        scoring=_if_mean_score_scorer,
        verbose=0,
        n_jobs=-1
    )
    grid.fit(Z_tr)

    best_if = grid.best_estimator_
    best_params = grid.best_params_

    # Predict on FINAL
    preds = best_if.predict(Z_fi)                 # {-1, +1}
    acc = float(np.mean(preds == y_fin))
    scores = -best_if.score_samples(Z_fi)         # anomaly-positive
    auc_fin = manual_auc(y_fin, scores, positive_label=-1)

    print(f"[CP+IF] FINAL AUC={auc_fin:.3f} | Acc={acc:.3f} | params={best_params}")

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(y_fin, preds, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"]).plot()
        plt.show()

    return acc, auc_fin

def cp_rank_search_isolation_forest(data_bundle):
    print('CP rank search (Isolation Forest)')
    startRank = 10; endRank = 385; step = 5
    rank_score = {}
    for rank in range(startRank, endRank, step):
        print('Rank:', rank)
        acc, auc = parafac_isolation_forest(rank, data_bundle, displayConfusionMatrix=False)
        rank_score[rank] = auc
        print('Accuracy', acc, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]

# Raw-pixel IsolationForest (no decomposition), using shared data path
def isolation_forest_anomaly(data_bundle, displayConfusionMatrix=False, random_state=42):
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Flatten + scale (fit on TRAIN)
    n_tr = X_train.shape[0]
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_train.reshape(n_tr, -1))

    n_fi = X_fin.shape[0]
    Z_fi = scaler.transform(X_fin.reshape(n_fi, -1))

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
    iso = IsolationForest()
    grid = GridSearchCV(iso, param_grid=param_grid, cv=3, scoring=_if_mean_score_scorer, n_jobs=-1, verbose=0)
    grid.fit(Z_tr)

    best_if = grid.best_estimator_
    preds = best_if.predict(Z_fi)
    acc = float(np.mean(preds == y_fin))
    scores = -best_if.score_samples(Z_fi)
    auc_fin = manual_auc(y_fin, scores, positive_label=-1)

    print(f"[IF(raw)] FINAL AUC={auc_fin:.3f} | Acc={acc:.3f} | params={grid.best_params_}")

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(y_fin, preds, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal']).plot()
        plt.show()

    return acc

# =============================================================================
# === Tucker + Isolation Forest ===============================================
# =============================================================================
def tucker_isolation_forests(rank, data_bundle, displayConfusionMatrix=False, random_state=42):
    """
    Tucker features (core + factors via extractFeatures) -> IsolationForest.
    """
    # Common split & standardization
    X_train, X_val, X_fin, _, y_fin, _, _ = get_splits(data_bundle, standardize=USE_BAND_STANDARDIZE)
    y_fin = np.asarray(y_fin, dtype=int)

    # Tucker decomposition per split
    n_tr, n_fi = X_train.shape[0], X_fin.shape[0]
    decomp_tr = buildTensor(X_train, rank, n_tr, isTuckerDecomposition=True)
    decomp_fi = buildTensor(X_fin,   rank, n_fi, isTuckerDecomposition=True)

    # Feature extraction + scaling (fit on TRAIN)
    Feat_tr = extractFeatures(decomp_tr, n_tr, isTuckerDecomposition=True)
    Feat_fi = extractFeatures(decomp_fi, n_fi, isTuckerDecomposition=True)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Feat_tr)
    Z_fi = scaler.transform(Feat_fi)

    # Hyperparameter tuning (unsupervised) on TRAIN only
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
    iso = IsolationForest()
    grid = GridSearchCV(iso, param_grid=param_grid, cv=3, scoring=_if_mean_score_scorer, n_jobs=-1, verbose=0)
    grid.fit(Z_tr)

    best_if = grid.best_estimator_
    preds = best_if.predict(Z_fi)
    acc = float(np.mean(preds == y_fin))
    scores = -best_if.score_samples(Z_fi)
    auc_fin = manual_auc(y_fin, scores, positive_label=-1)

    print(f"[Tucker+IF] FINAL AUC={auc_fin:.3f} | Acc={acc:.3f} | params={grid.best_params_}")

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(y_fin, preds, labels=[-1, 1])
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"]).plot()
        plt.show()

    return acc, auc_fin

def tucker_rank_search_isolation_forest(data_bundle):
    print('Tucker rank search (Isolation Forest)')
    rankSet = sorted({5, 16, 32, 64})
    rank_score = {}
    for i in rankSet:
        for j in rankSet:
            for k in sorted({5, 16}):  # keep k modest (band mode)
                r = (i, j, k)
                print('Rank:', r)
                acc, auc = tucker_isolation_forests(r, data_bundle)
                rank_score[r] = auc
                print('Accuracy:', acc, 'AUC', auc)
    print('AUC Rank score', rank_score)
    bestRank = max(rank_score, key=rank_score.get)
    return bestRank, rank_score[bestRank]


# ====== Entry (reads once, then passes data to pipelines) ====================
data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42)

if enable_cp_oc_svm:
    if no_decomposition:
        one_class_svm()
    else:
        if use_predefined_rank == False:
            bestRank, bestAUC = cp_rank_search_one_class_svm(data_bundle)
            print('Best Rank for CP with One Class SVM', bestRank, bestAUC)
        else:
            print('Running best rank CP OC-SVM')
            bestRank = 24
            parafac_OC_SVM(bestRank, data_bundle, True)

if enable_pca_oc_svm:
    print('Running PCA + OC-SVM on raw pixels')
    for nComponent in np.sort([16, 32, 48, 64, 128, 256, 512]):
        pca_OC_SVM(data_bundle, n_components=nComponent, whiten=PCA_WHITEN, randomized=PCA_RANDOMIZED, displayConfusionMatrix=False)

if enable_tucker_oc_svm:
    if use_predefined_rank == False:
        bestRank, bestAUC = tucker_rank_search_one_class_svm(data_bundle)
        print('Tucker Best Rank One Class SVM', bestRank, bestAUC)
    else:
        print('Running best rank Tucker with one-class SVM')
        rank = (5, 5, 35)
        accuracy = tucker_one_class_svm(rank, data_bundle, True)
        print('Tucker OC-SVM accuracy @ predefined rank:', accuracy)

if enable_cp_autoencoder:
    if no_decomposition:
        autoencoder(data_bundle)
    else:
        if use_predefined_rank == False:
            bestRank, bestAUC = cp_rank_search_autoencoder(data_bundle)
            print('Best Rank for CP with autoencoder', bestRank, bestAUC)
        else:
            print('Running best rank CP with autoencoder')
            bestRank = 85
            parafac_autoencoder(bestRank, factor=2, bottleneck=32, data_bundle=data_bundle)

if enable_tucker_autoencoder:
    if use_predefined_rank == False:
        bestRank, bestAUC = tucker_rank_search_autoencoder(data_bundle)
        print('Best Rank Tucker with autoencoder', bestRank, bestAUC)
    else:
        print('Running best rank Tucker with autoencoder')
        rank = (5, 5, 35)
        factor = 1
        accuracy = tucker_neural_network_autoencoder(rank, factor, 16, data_bundle, True)

if enable_cp_isolation_forest:
    if no_decomposition:
        print('Isolation Forest (raw pixels)')
        accuracy = isolation_forest_anomaly(data_bundle)
        print('IsolationForest (raw) accuracy', accuracy)
    else:
        if use_predefined_rank == False:
            bestRank, bestAUC = cp_rank_search_isolation_forest(data_bundle)
            print('Best Rank for CP with Isolation Forest', bestRank, bestAUC)
        else:
            print('Running best rank CP with Isolation Forest')
            bestRank = 24
            parafac_isolation_forest(bestRank, data_bundle, True)

if enable_tucker_isolation_forest:
    if use_predefined_rank == False:
        bestRank, bestAUC = tucker_rank_search_isolation_forest(data_bundle)
        print('Best Rank Tucker with Isolation Forest', bestRank, bestAUC)
    else:
        print('Running best rank Tucker with Isolation Forest')
        rank = (5, 65, 5)
        accuracy = tucker_isolation_forests(rank, data_bundle, True)

