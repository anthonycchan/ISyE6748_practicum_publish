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

no_decomposition = False  # set to False to run CP-based pipeline

# Optional: standardize bands using TRAIN stats (recommended)
USE_BAND_STANDARDIZE = True

# ====== Single-component selection toggle ====================================
# When True, auto-select the best single CP component on VAL and train OC-SVM on that 1-D feature.
USE_SINGLE_CP_COMPONENT = True

# ====== Dataset reduction controls ===========================================
REDUCE_DATASETS = False
REDUCE_TRAIN_N = 1500
REDUCE_VAL_N = 200
REDUCE_TEST_TYP_N = 200
REDUCE_TEST_ANO_N = 200
REDUCE_SEED = 123  # reproducible subsets
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
    if n_keep is None or n_keep >= n_total:
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
def decompose_tensor_tucker(tensor, rank):
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

    X_train = np.asarray(data_bundle["X_train"], dtype=np.float32)
    X_val   = np.asarray(data_bundle["X_val"],   dtype=np.float32)
    X_fin   = np.asarray(data_bundle["X_fin"],   dtype=np.float32)
    y_val   = data_bundle.get("y_val", None)
    y_fin   = np.asarray(data_bundle["y_fin"],   dtype=int)

    # ---- Standardize with TRAIN stats only
    if USE_BAND_STANDARDIZE:
        mu_b, sig_b = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu_b, sig_b)
        X_val   = apply_band_standardizer(X_val,   mu_b, sig_b)
        X_fin   = apply_band_standardizer(X_fin,   mu_b, sig_b)

    # Basic sanity
    sanity_report(X_train, "TRAIN")
    sanity_report(X_val,   "VAL")
    sanity_report(X_fin,   "FINAL")

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("[sanity] Cleaning TRAIN tiles ..."); X_train = clean_tiles(X_train)
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("[sanity] Cleaning VAL tiles ...");   X_val = clean_tiles(X_val)
    if np.isnan(X_fin).any() or np.isinf(X_fin).any():
        print("[sanity] Cleaning FINAL tiles ..."); X_fin = clean_tiles(X_fin)

    # ---- GLOBAL CP on TRAIN
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state,
        max_train_samples=cp_basis_max_train_samples
    )

    # ---- PROJECT VAL + FINAL onto same basis
    Ginv = precompute_cp_projection(A, B, C)
    H_val = project_cp_coeffs(X_val, A, B, C, Ginv=Ginv)
    H_fin = project_cp_coeffs(X_fin, A, B, C, Ginv=Ginv)

    # ---- Scale
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    # ---- Choose features: single component (auto) or full H (with/without PCA)
    if USE_SINGLE_CP_COMPONENT:
        k_best = select_best_single_cp_component(Htr_s, Hval_s, y_val)
        Htr_w  = Htr_s[:, [k_best]]
        Hval_w = Hval_s[:, [k_best]]
        Hfin_w = Hfin_s[:, [k_best]]
        feat_dim = 1
        print(f"[CP+OCSVM] Using single CP component k={k_best} (auto-selected)")
    else:
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
    return acc_opt

# ====== Optional raw-pixel OC-SVM (unchanged) ================================
def ocsvm_raw_geography(displayConfusionMatrix=False):
    # === Speed toggles (safe defaults) =======================================
    _USE_PCA = False           # NEW: set True to enable PCA dimensionality reduction
    _PCA_COMPONENTS = 256      # NEW: 128–512 is a good range
    _PARALLEL_GRID = True      # NEW: parallelize the param grid
    _SVM_CACHE_MB = 1024       # NEW: increase libsvm kernel cache (MB)
    _SVM_TOL = 1e-2            # NEW: slightly looser tolerance for faster convergence
    _SVM_MAX_ITER = -1         # NEW: keep default unlimited; set e.g. 1_00_000 if needed
    _USE_NYSTROEM = False      # NEW: RBF approximation -> OCSVM linear (set True to try)
    _NY_COMPONENTS = 512       # NEW: 256–1024 typical; tradeoff speed/accuracy

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import ParameterGrid
    from sklearn import metrics
    import matplotlib.pyplot as plt

    # Optional deps only used if toggled
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

    scaler = StandardScaler(copy=False)  # NEW: avoid an extra copy
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    # === Optional PCA reduction (massive speedup in high-D) ==================
    if _USE_PCA:
        pca = PCA(n_components=_PCA_COMPONENTS, svd_solver='randomized', random_state=REDUCE_SEED)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled  = pca.transform(X_test_scaled)

    # === Optional Nystroem RBF approximation (then use linear OCSVM) =========
    # NOTE: This path ignores 'kernel' in the grid (it uses linear) and maps gamma via Nystroem.
    # It’s much faster; try it if the exact RBF OCSVM is too slow.
    def fit_eval_linear_on_map(gamma):
        mapper = Nystroem(gamma=gamma, n_components=_NY_COMPONENTS, random_state=REDUCE_SEED)
        Z_train = mapper.fit_transform(X_train_scaled)
        Z_test  = mapper.transform(X_test_scaled)
        return Z_train, Z_test

    # === Param grid ==========================================================
    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    best_acc = -1; best_model = None; best_params = None
    y_true = np.asarray(true_labels)

    def _train_eval(params):
        # NEW: inject faster SVM knobs
        p = dict(params)
        p.setdefault('cache_size', _SVM_CACHE_MB)
        p.setdefault('tol', _SVM_TOL)
        p.setdefault('max_iter', _SVM_MAX_ITER)

        if _USE_NYSTROEM and p.get('kernel', 'rbf') == 'rbf' and p['gamma'] not in ('scale', 'auto'):
            # Map to low-D then use linear OCSVM
            Z_train, Z_test = fit_eval_linear_on_map(p['gamma'])
            p_lin = dict(nu=p['nu'], kernel='linear', cache_size=_SVM_CACHE_MB, tol=_SVM_TOL, max_iter=_SVM_MAX_ITER)
            model = OneClassSVM(**p_lin).fit(Z_train)
            preds = model.predict(Z_test)
        else:
            model = OneClassSVM(**p).fit(X_train_scaled)
            preds = model.predict(X_test_scaled)

        acc = float(np.mean(preds == y_true))
        return acc, params, model

    # === Run grid (parallel if enabled) ======================================
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

    # === Pick best ===========================================================
    for acc, params, model in results:
        if acc > best_acc:
            best_acc, best_params, best_model = acc, params, model

    print(f"Best accuracy (grid @ default threshold 0): {best_acc:.3f} with params: {best_params}")

    # === Scoring/thresholds on the best model ================================
    if _USE_NYSTROEM and best_params.get('kernel', 'rbf') == 'rbf' and best_params['gamma'] not in ('scale', 'auto'):
        Z_train, Z_test = fit_eval_linear_on_map(best_params['gamma'])  # recompute map for best gamma
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
    startRank = 10; endRank = 260; step = 5  # tighter range for speed
    rank_accuracy = {}
    for rank in range(startRank, endRank, step):
        print('Rank:', rank)
        accuracy = parafac_OC_SVM(rank, data_bundle, use_pca_whiten=False)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# ====== Entry (reads once, then passes data to CP pipeline) ==================
if enable_cp_oc_svm:
    if no_decomposition:
        one_class_svm()
    else:
        data_bundle = prepare_data_once(val_fraction=VAL_FRACTION, random_state=42)
        if use_predefined_rank == False:
            bestRank, bestAccuracy = cp_rank_search_one_class_svm(data_bundle)
            print('Best Rank for CP with One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank CP OC-SVM')
            bestRank = 24
            parafac_OC_SVM(bestRank, data_bundle, True)
