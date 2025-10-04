# ====== Imports ===============================================================
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed

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

random.seed(1)

# ====== Paths & toggles (as in your project) =================================
train_data        = "Data/Reduced/Lean/train"        # typical only
validation_data   = "Data/Reduced/Lean/validation"   # typical only
test_typical_data = "Data/Reduced/Lean/test_typical" # typical
test_anomaly_data = "Data/Reduced/Lean/test_novel"   # novel

use_predefined_rank = False
enable_tucker_oc_svm = False
enable_tucker_autoencoder = False
enable_tucker_random_forest = False
enable_cp_oc_svm = True        # <-- main one you run
enable_cp_autoencoder = False
enable_cp_random_forest = False

no_decomposition = False  # set to False to run CP-based pipeline

# Optional: standardize bands using TRAIN stats (recommended)
USE_BAND_STANDARDIZE = True

# ====== NEW: TensorLy backend + numeric-stability helpers ====================
# Choose backend: "pytorch" (GPU if CUDA available) or "numpy" (CPU).
TL_BACKEND = "pytorch"   # change to "numpy" to force CPU
DEVICE = "cpu"

USE_GPU_CP = True        # new: toggle GPU for CP fit + projection

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

# ====== Your original I/O ====================================================
def readData(directory):
    directory = os.fsencode(directory)
    filelist = os.listdir(directory)
    numFiles = len(filelist)
    data_set = np.ones([numFiles, 64, 64, 6], dtype=np.float32)
    true_labels = []
    i = 0
    for file in filelist:
        filename = os.fsdecode(file)
        true_labels.append(1)
        img_array = np.load(os.fsdecode(directory) + "/" + filename).astype(np.float32)
        img_array = img_array / 255.0
        data_set[i, :, :, :] = img_array
        i += 1
    return data_set, true_labels

def readData_test(typical_dir, anomaly_dir):
    typical_dir = os.fsdecode(typical_dir)
    anomaly_dir = os.fsdecode(anomaly_dir)
    typical_files = sorted([f for f in os.listdir(typical_dir) if f.endswith(".npy")])
    anomaly_files = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(".npy")])
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

# ====== DROP-IN replacements for CP/Tucker decompose =========================
def decompose_tensor_tucker(tensor, rank):
    """
    Robust Tucker:
    - float64 on GPU
    - falls back to HOSVD and then CPU if needed
    Returns: (core, factors) as numpy arrays.
    """
    ranks = rank
    try:
        Xb = _to_backend(tensor, use_float64=True)
        core, factors = _tl_tucker(Xb, ranks=ranks, init="svd", n_iter_max=500, tol=1e-6)
        core_np = _to_numpy(core)
        facs_np = [ _to_numpy(Fm) for Fm in factors ]
        if np.all(np.isfinite(core_np)) and all(np.all(np.isfinite(Fm)) for Fm in facs_np):
            return core_np, facs_np
    except Exception:
        pass
    try:
        Xb = _to_backend(tensor, use_float64=True)
        core, factors = _tl_tucker(Xb, ranks=ranks, init="svd", n_iter_max=1, tol=0)
        core_np = _to_numpy(core)
        facs_np = [ _to_numpy(Fm) for Fm in factors ]
        if np.all(np.isfinite(core_np)) and all(np.all(np.isfinite(Fm)) for Fm in facs_np):
            return core_np, facs_np
    except Exception:
        pass
    try:
        old = tl.get_backend()
        tl.set_backend("numpy")
        core, factors = _tl_tucker(tensor.astype(np.float32), ranks=ranks, init="svd", n_iter_max=500, tol=1e-6)
        tl.set_backend(old)
        return core.astype(np.float32), [Fm.astype(np.float32) for Fm in factors]
    except Exception:
        raise RuntimeError("Tucker failed on all backends for this tile.")

def decompose_tensor_parafac(tensor, rank):
    """
    Robust CP-ALS:
    - float64 on GPU
    - CPU fallback if needed
    Returns: factors (list of numpy arrays).
    """
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

# ====== Your feature extractors (unchanged) =================================
def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

# ====== Your tensor build/extract pipelines (unchanged) =====================
def buildTensor(X, rank, num_sets, isTuckerDecomposition=True, ordered=False):
    if ordered:
        decomposed_data = [None] * num_sets
        with ThreadPoolExecutor() as executor:
            if isTuckerDecomposition:
                future_to_index = {executor.submit(decompose_tensor_tucker, X[i], rank): i for i in range(num_sets)}
            else:
                future_to_index = {executor.submit(decompose_tensor_parafac, X[i], rank): i for i in range(num_sets)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    decomposed_data[idx] = future.result()
                except Exception as e:
                    print(f"Error in tensor decomposition at index {idx}: {e}")
    else:
        with ThreadPoolExecutor() as executor:
            if isTuckerDecomposition:
                decomposed_data = list(executor.map(lambda i: decompose_tensor_tucker(X[i], rank), range(num_sets)))
            else:
                decomposed_data = list(executor.map(lambda i: decompose_tensor_parafac(X[i], rank), range(num_sets)))
    return decomposed_data

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

# ====== Evaluation helpers (unchanged) ======================================
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

# ====== (Optional) CP/Tucker viz helper (unchanged) =========================
def visualize_cp(decomposed_list, rank, sample_index=0, img_side1=64, img_side2=64, max_components=3, verbose=True):
    if not isinstance(decomposed_list, (list, tuple)) or len(decomposed_list) == 0:
        if verbose: print("visualize_cp: decomposed_list is empty or not a sequence; nothing to plot.")
        return False
    idx = min(max(sample_index, 0), len(decomposed_list) - 1)
    item = decomposed_list[idx]
    factors = None; core = None
    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
        core, factors = item[0], item[1]
    elif isinstance(item, (list, tuple)):
        if len(item) > 0 and hasattr(item[0], 'ndim'):
            factors = item
        else:
            if verbose: print("visualize_cp: sequence item doesn't look like CP/Tucker factors; skipping.")
            return False
    else:
        if verbose: print("visualize_cp: unsupported element type in list; skipping.")
        return False
    try:
        F = [np.asarray(Fm) for Fm in factors]
    except Exception as e:
        if verbose: print("visualize_cp: could not convert factors to arrays:", e)
        return False
    if len(F) == 0 or any((Fm.ndim != 2 or Fm.shape[1] < 1) for Fm in F):
        if verbose: print("visualize_cp: malformed factor list; skipping.")
        return False
    R = min(int(rank), min(Fm.shape[1] for Fm in F))
    if R < 1:
        if verbose: print("visualize_cp: rank has no columns to plot; skipping.")
        return False
    if verbose:
        shapes = [Fm.shape for Fm in F]
        print(f"visualize_cp: sample={idx}, mode shapes={shapes}, using R={R}")
    for m, Fm in enumerate(F):
        try:
            plt.figure(figsize=(7, 3.5))
            for r in range(R):
                plt.plot(Fm[:, r], label=f'Comp {r+1}')
            title = f'Mode {m+1} Factor Loadings' + (' (Tucker)' if core is not None else ' (CP)')
            plt.title(title)
            if Fm.shape[1] > 1:
                plt.legend(loc='upper right', ncols=2 if R >= 4 else 1, fontsize=8)
            plt.tight_layout(); plt.show()
        except Exception as e:
            if verbose: print(f"visualize_cp: line-plot for mode {m+1} skipped:", e)
    heatmaps_done = False
    if len(F) >= 2 and F[0].shape[0] == img_side1 and F[1].shape[0] == img_side2:
        A, B = F[0], F[1]
        for r in range(min(R, max_components)):
            try:
                comp_img = np.outer(A[:, r], B[:, r]).reshape(img_side1, img_side2)
                plt.figure(figsize=(4.2, 4.2)); plt.imshow(comp_img, cmap='viridis'); plt.colorbar()
                lbl = 'Tucker' if core is not None else 'CP'
                plt.title(f'{lbl} Spatial Map: Component {r+1} ({img_side1}×{img_side2})')
                plt.tight_layout(); plt.show(); heatmaps_done = True
            except Exception as e:
                if verbose: print(f"visualize_cp: heatmap for component {r+1} skipped:", e)
    if not heatmaps_done and verbose:
        print("visualize_cp: no 2D heatmaps (first two mode sizes not", img_side1, "and", img_side2, ").")
    return True

# ====== NEW: Global CP basis + projection for OC-SVM ========================
def fit_global_cp_basis(X_train, rank, random_state=42, max_train_samples=None, use_gpu=USE_GPU_CP):
    """
    Fit one global CP model to TRAIN tensor (N,64,64,6).
    Returns:
      (A,B,C) as np.float32 and H_train as np.float32 (N,R).
    If subsampling is used, H_train is recomputed for the full TRAIN via GPU projection.
    """
    import numpy as _np
    N = X_train.shape[0]
    X_in = X_train
    if max_train_samples is not None and N > max_train_samples:
        idx = np.random.RandomState(123).choice(N, size=max_train_samples, replace=False)
        X_in = X_in[idx]

    # GPU branch (TensorLy+PyTorch) ---------------------------
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

                # Convert basis to NumPy for downstream use (sklearn expects NumPy)
                A = A_t.detach().cpu().numpy().astype(_np.float32)
                B = B_t.detach().cpu().numpy().astype(_np.float32)
                C = C_t.detach().cpu().numpy().astype(_np.float32)

                if X_in.shape[0] != X_train.shape[0]:
                    # Recompute H for the FULL training set via GPU projection
                    H_full = project_cp_coeffs_torch(X_train, A, B, C, device="cuda")
                    return (A, B, C), H_full.astype(_np.float32)
                else:
                    H_np = H_t.detach().cpu().numpy().astype(_np.float32)
                    return (A, B, C), H_np
        # fall through to CPU if no CUDA

    # CPU fallback (your original NumPy path) -----------------
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

def ocsvm_gamma_grid_for_dim(d):
    base = 1.0 / max(d, 1)
    return [base * t for t in (0.1, 0.3, 1.0, 3.0, 10.0)]

# ====== UPDATED: CP + OC-SVM with separate VAL dir (typical-only) ===========
def parafac_OC_SVM(rank, displayConfusionMatrix=False, use_pca_whiten=True,
                   val_fraction=0.5, random_state=42, cp_basis_max_train_samples=None):
    """
    Pipeline:
      1) Fit a single global CP basis on TRAIN (optionally subsampled).
      2) Project VALIDATION (typical-only if provided) and FINAL (test) onto basis.
      3) Select OC-SVM params using VALIDATION normals only (one-class criterion).
      4) Evaluate on FINAL with anomalies present.
    """
    start_time = time.time()
    rng = np.random.RandomState(random_state)

    # ---- 1) Load data
    X_train, _ = readData(train_data)                             # typical-only
    X_pool,  y_pool  = readData_test(test_typical_data, test_anomaly_data)  # labeled test pool
    y_pool = np.asarray(y_pool, int)

    # If validation dir exists with .npy files, use it as VAL (typical-only).
    use_separate_val = _dir_has_npy(validation_data)
    if use_separate_val:
        X_val_typ, _ = readData(validation_data)                  # typical-only VAL
        X_fin, y_fin = X_pool, y_pool                             # FINAL = whole test pool
        print(f"[VAL] Using separate validation dir (typical-only): {validation_data} (N={X_val_typ.shape[0]})")
    else:
        # Fall back to prior stratified split from test pool
        print("[VAL] Separate validation dir missing/empty; falling back to stratified VAL/FINAL split from test pool.")
        idx_all = np.arange(len(y_pool)); rng.shuffle(idx_all)
        X_pool = X_pool[idx_all]; y_pool = y_pool[idx_all]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction, random_state=random_state)
        val_idx, final_idx = next(sss.split(np.zeros_like(y_pool), y_pool))
        X_val_typ, y_val = X_pool[val_idx], y_pool[val_idx]       # here VAL has both classes
        X_fin, y_fin = X_pool[final_idx], y_pool[final_idx]
        # We'll detect below whether y_val has anomalies and use AUC if available.

    # ---- 2) Optional: train-only band standardization
    if USE_BAND_STANDARDIZE:
        mu_b, sig_b = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu_b, sig_b)
        X_fin   = apply_band_standardizer(X_fin,   mu_b, sig_b)
        X_val_typ = apply_band_standardizer(X_val_typ, mu_b, sig_b)

    # Basic sanity
    sanity_report(X_train, "TRAIN")
    sanity_report(X_val_typ, "VAL (typical-only read)")
    sanity_report(X_fin,   "FINAL (test)")

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("[sanity] Cleaning TRAIN tiles ..."); X_train = clean_tiles(X_train)
    if np.isnan(X_val_typ).any() or np.isinf(X_val_typ).any():
        print("[sanity] Cleaning VAL tiles ...");   X_val_typ = clean_tiles(X_val_typ)
    if np.isnan(X_fin).any() or np.isinf(X_fin).any():
        print("[sanity] Cleaning FINAL tiles ..."); X_fin = clean_tiles(X_fin)

    # ---- 3) GLOBAL CP on TRAIN
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state,
        max_train_samples=cp_basis_max_train_samples
    )

    # ---- 4) PROJECT VAL (typical-only) + FINAL onto same basis
    Ginv = precompute_cp_projection(A, B, C)
    H_val = project_cp_coeffs(X_val_typ, A, B, C, Ginv=Ginv)
    H_fin = project_cp_coeffs(X_fin,     A, B, C, Ginv=Ginv)

    # ---- 5) Scale + optional PCA whitening (fit on TRAIN only)
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    if use_pca_whiten:
        pca   = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Htr_w  = pca.fit_transform(Htr_s)
        Hval_w = pca.transform(Hval_s)
        Hfin_w = pca.transform(Hfin_s)
        feat_dim = Htr_w.shape[1]
    else:
        pca = None
        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s
        feat_dim = Htr_w.shape[1]

    # ---- 6) OC-SVM hyperparams (γ scaled to 1/d)
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(feat_dim),
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    # Determine selection mode:
    # - If we fell back and VAL has both classes, we can still use AUC.
    # - Otherwise (separate validation typical-only), use one-class criterion.
    use_auc_on_val = (not use_separate_val) and (np.isin(-1, y_val).any() and np.isin(1, y_val).any()) if 'y_val' in locals() else False

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
                # AUC selection (original behavior) when VAL has both classes
                auc = manual_auc(y_val, s_val, positive_label=-1)
                if not np.isfinite(auc):
                    continue
                score_tuple = (-auc,)  # minimize negative AUC == maximize AUC
                aux = f"AUC={auc:.3f}"
            else:
                # One-class selection on typical-only VAL:
                # 1) minimize FP rate at default cutoff (s_val >= 0)
                # 2) then minimize 95th percentile (robust worst-case normal)
                # 3) then minimize mean score
                fp_rate = float((s_val >= 0.0).mean())
                p95 = float(np.percentile(s_val, 95))
                mean_s = float(np.mean(s_val))
                score_tuple = (fp_rate, p95, mean_s)  # lexicographic minimization
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
            # Degenerate final fallback
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

    # ---- 7) FINAL evaluation on test pool
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
    return acc_opt

# ====== (Optional) raw-pixel OC-SVM (unchanged) =============================
def ocsvm_raw_geography(displayConfusionMatrix=False):
    X_train, _ = readData(train_data)
    X_test, true_labels = readData_test(test_typical_data, test_anomaly_data)
    if USE_BAND_STANDARDIZE:
        mu, sig = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu, sig)
        X_test  = apply_band_standardizer(X_test,  mu, sig)
    n_train = X_train.shape[0]; n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1); X_test_flat = X_test.reshape(n_test, -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)
    param_grid = {'nu': [0.05, 0.1, 0.2], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid']}
    best_acc = -1; best_model = None; best_params = None
    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params); model.fit(X_train_scaled)
        preds = model.predict(X_test_scaled)  # +1 normal, -1 anomaly
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc; best_model = model; best_params = params
    print(f"Best accuracy (grid @ default threshold 0): {best_acc:.3f} with params: {best_params}")
    predictions_default = best_model.predict(X_test_scaled)
    scores_normal = best_model.decision_function(X_test_scaled).ravel()
    scores_anom = -scores_normal
    accuracy_default = np.mean(predictions_default == true_labels)
    print("Accuracy @ default cutoff (0):", float(accuracy_default))
    auc_manual = manual_auc(true_labels, scores_anom, positive_label=-1)
    print("ROC AUC (manual, anomaly-positive):", auc_manual)
    th_opt, acc_opt = _pick_threshold_max_accuracy(true_labels, scores_anom, positive_label=-1)
    print(f"Chosen threshold (max-accuracy on ROC scores): {th_opt:.6f}  |  Accuracy @ chosen: {acc_opt:.3f}")
    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, np.where(scores_anom >= th_opt, -1, 1), labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot(); plt.show()
    return acc_opt, best_params

def one_class_svm():
    print('One Class SVM')
    accuracy, param = ocsvm_raw_geography(False)
    print('One class SVM best accuracy:', accuracy, 'param:', param)

def cp_rank_search_one_class_svm():
    print('CP rank search One Class SVM')
    startRank = 10; endRank = 260; step = 5  # tighter range for speed
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_OC_SVM(rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# ====== Entry (same shape as your original) =================================
if enable_cp_oc_svm:
    if no_decomposition:
        one_class_svm()
    else:
        if use_predefined_rank == False:
            bestRank, bestAccuracy = cp_rank_search_one_class_svm()
            print('Best Rank for CP with One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank CP OC-SVM')
            bestRank = 24
            parafac_OC_SVM(bestRank, True)
