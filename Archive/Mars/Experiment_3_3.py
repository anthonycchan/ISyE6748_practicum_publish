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
train_data = "Data/Reduced/Lean/train"
validation_data = "Data/Reduced/Lean/validation"
test_typical_data = "Data/Reduced/Lean/test_typical"
test_anomaly_data = "Data/Reduced/Lean/test_novel"

use_predefined_rank = False
enable_tucker_oc_svm = False
enable_tucker_autoencoder = False
enable_tucker_random_forest = False
enable_cp_oc_svm = True        # <-- main one you run
enable_cp_autoencoder = False
enable_cp_random_forest = False
no_decomposition = False

# Optional: standardize bands using TRAIN stats (recommended)
USE_BAND_STANDARDIZE = True

# ====== NEW: TensorLy backend + numeric-stability helpers ====================
# Choose backend: "pytorch" (GPU if CUDA available) or "numpy" (CPU).
TL_BACKEND = "pytorch"   # change to "numpy" to force CPU
DEVICE = "cpu"

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

# (Your quick visualization calls — optional)
# X, _ = readData(train_data); displayImages(X, {0, 10, 100})
# X, _ = readData(test_anomaly_data)
# displayImages(X, {0, 10, 20})

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

# data_set, _ = readData(train_data); plot_signals(data_set, 10)
# plt.show()

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
def fit_global_cp_basis(X_train, rank, random_state=42, max_train_samples=None):
    """
    Fit one global CP model to TRAIN tensor of shape (N,64,64,6).
    Returns:
      basis = (A, B, C) with shapes (64,R), (64,R), (6,R)
      H_train: (N,R) sample-mode coefficients (weights folded in)
    Notes:
      - If max_train_samples is not None, subsample TRAIN to that size for basis fit.
      - Uses current TL backend (float64 on GPU via _to_backend path not strictly needed here
        because we call NumPy CP for stability on big 4-way tensors).
    """
    import numpy as _np

    X_in = X_train
    N = X_in.shape[0]
    if max_train_samples is not None and N > max_train_samples:
        idx = np.random.RandomState(123).choice(N, size=max_train_samples, replace=False)
        X_in = X_in[idx]

    # Prefer CPU/NumPy for the big 4-way fit (more stable memory-wise)
    old_backend = tl.get_backend()
    try:
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            X_in.astype(_np.float32), rank=rank, init="random",
            n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
        )
        # factors: [H (N'×R), A (64×R), B (64×R), C (6×R)]
        H, A, B, C = [ _np.asarray(F) for F in factors ]
        lam = _np.asarray(weights)  # (R,)
        H = H * lam[None, :]        # fold weights into sample coeffs
        # If we subsampled TRAIN, refit H for full TRAIN on fixed A,B,C below:
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

# ====== UPDATED: CP + OC-SVM uses GLOBAL CP BASIS ============================
def parafac_OC_SVM(rank, displayConfusionMatrix=False, use_pca_whiten=True,
                   val_fraction=0.5, random_state=42, cp_basis_max_train_samples=None):
    """
    Same API as before, but now:
      1) Fits a **single global CP basis** on TRAIN (optionally subsampled).
      2) Projects VAL/FINAL onto that basis to get comparable R-dim features (H).
      3) Selects OC-SVM params on VAL by AUC; evaluates on FINAL.

    Returns:
      acc_opt on FINAL (for continuity with your code), after threshold tuning.
    """
    start_time = time.time()
    rng = np.random.RandomState(random_state)

    # ---- 1) Load data
    X_train, _ = readData(train_data)                             # typical-only
    X_pool,  y_pool  = readData_test(test_typical_data, test_anomaly_data)  # labeled pool
    y_pool = np.asarray(y_pool, int)

    # ---- 2) Optional: train-only band standardization
    if USE_BAND_STANDARDIZE:
        mu_b, sig_b = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu_b, sig_b)
        X_pool  = apply_band_standardizer(X_pool,  mu_b, sig_b)

    # Basic sanity
    sanity_report(X_train, "TRAIN std" if USE_BAND_STANDARDIZE else "TRAIN raw")
    sanity_report(X_pool,  "POOL std"  if USE_BAND_STANDARDIZE else "POOL raw")
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("[sanity] Cleaning TRAIN tiles ..."); X_train = clean_tiles(X_train)
    if np.isnan(X_pool).any() or np.isinf(X_pool).any():
        print("[sanity] Cleaning POOL tiles ...");  X_pool  = clean_tiles(X_pool)

    # ---- 3) Split pool into VAL/FINAL (stratified)
    idx_all = np.arange(len(y_pool)); rng.shuffle(idx_all)
    X_pool = X_pool[idx_all]; y_pool = y_pool[idx_all]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction, random_state=random_state)
    val_idx, final_idx = next(sss.split(np.zeros_like(y_pool), y_pool))
    X_val, y_val = X_pool[val_idx], y_pool[val_idx]
    X_fin, y_fin = X_pool[final_idx], y_pool[final_idx]

    # ---- 4) GLOBAL CP on TRAIN
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state,
        max_train_samples=cp_basis_max_train_samples
    )

    # ---- 5) PROJECT VAL/FINAL onto same basis
    Ginv = precompute_cp_projection(A, B, C)
    H_val = project_cp_coeffs(X_val, A, B, C, Ginv=Ginv)
    H_fin = project_cp_coeffs(X_fin, A, B, C, Ginv=Ginv)

    # ---- 6) Scale + optional PCA whitening
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    if use_pca_whiten:
        pca   = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Htr_w = pca.fit_transform(Htr_s)
        Hval_w = pca.transform(Hval_s)
        Hfin_w = pca.transform(Hfin_s)
        feat_dim = Htr_w.shape[1]
    else:
        pca = None
        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s
        feat_dim = Htr_w.shape[1]

    # ---- 7) OC-SVM hyperparams (γ scaled to 1/d)
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ocsvm_gamma_grid_for_dim(feat_dim),
        "nu":     [0.01, 0.02, 0.05, 0.1, 0.2]
    }

    # ---- 8) Pick params on VAL by **AUC** (higher is better)
    best_auc = -1.0
    best_model = None
    best_params = None
    for params in ParameterGrid(param_grid):
        try:
            model = OneClassSVM(**params).fit(Htr_w)
            s_val = -model.decision_function(Hval_w).ravel()  # higher => more anomalous
            auc = manual_auc(y_val, s_val, positive_label=-1)
            if np.isfinite(auc) and auc > best_auc:
                best_auc, best_model, best_params = auc, model, params
        except Exception:
            continue

    print(f"[CP+OCSVM] VAL AUC={best_auc:.3f} with params={best_params}")

    # ---- 9) FINAL evaluation
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
    startRank = 10; endRank = 90; step = 5  # tighter range for speed
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
