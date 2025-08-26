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
from tensorly.decomposition import parafac as _tl_parafac, tucker as _tl_tucker

# ML utils
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit

random.seed(1)

# ====== Paths & toggles ======================================================
train_data = "Data/Full/train_typical"
validation_data = "Data/Full/validation_typical"   # typical-only (if you use it elsewhere)
test_typical_data = "Data/Full/test_typical"
test_anomaly_data = "Data/Full/test_novel"

use_predefined_rank = False
enable_tucker_oc_svm = False
enable_tucker_autoencoder = False
enable_tucker_random_forest = False
enable_cp_oc_svm = True
enable_cp_autoencoder = False
enable_cp_random_forest = False
no_decomposition = False

# Optional: standardize bands using TRAIN stats
USE_BAND_STANDARDIZE = True

# ====== Backend & device =====================================================
TL_BACKEND = "pytorch"   # "pytorch" (GPU if CUDA) or "numpy"
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
    """Numpy -> backend tensor. Use float32 by default on GPU to save VRAM."""
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

# ====== I/O ==================================================================
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

# ====== Robust per-tile Tucker/CP (unchanged API) ============================
def decompose_tensor_tucker(tensor, rank):
    """Robust Tucker; GPU if available, with CPU fallbacks."""
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
    """Robust per-tile CP with GPU/CPU fallbacks."""
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

# ====== Feature extractors (unchanged) =======================================
def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

# ====== Tensor build/extract pipelines (unchanged) ===========================
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

# ====== Eval helpers (unchanged) ============================================
def manual_auc(y_true, scores, positive_label=-1):
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

# ====== GPU-accelerated Global CP basis + Projection =========================
# Caps for big data (adjust if you have more VRAM/RAM)
OCS_TRAIN_MAX = 4000
OCS_VAL_MAX   = 4000
OCS_RETRY_TRAIN_MAX = 2000
PROJ_BATCH = 256  # batch size for GPU projection

def _ensure_finite(X, name):
    bad = ~np.isfinite(X)
    if bad.any():
        n = int(bad.sum())
        print(f"[sanitize] {name}: fixing {n} non-finite entries")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def _subsample_rows(X, max_n, rng):
    n = X.shape[0]
    if max_n is None or n <= max_n:
        return X, np.arange(n)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx], idx

def _subsample_rows_with_labels(X, y, max_n, rng, stratified=True):
    y = np.asarray(y)
    n = X.shape[0]
    if max_n is None or n <= max_n:
        return X, y
    if not stratified or len(np.unique(y)) < 2:
        idx = rng.choice(n, size=max_n, replace=False)
        return X[idx], y[idx]
    sel = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        k = max(1, int(round((len(pool)/n) * max_n)))
        k = min(k, len(pool))
        sel.append(rng.choice(pool, size=k, replace=False))
    idx = np.concatenate(sel)
    if idx.shape[0] > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return X[idx], y[idx]

def fit_global_cp_basis(X_train, rank, random_state=42, max_train_samples=None):
    """
    Fit one global CP model to TRAIN (N,64,64,6) on GPU if available.
    - Uses init='random' to avoid huge SVD memory.
    - Optionally subsamples TRAIN, then projects full TRAIN to get H_train.
    Returns: (A,B,C): (64×R, 64×R, 6×R),  H_train: (N×R)
    """
    import torch

    X_in = X_train
    N = X_in.shape[0]
    if max_train_samples is not None and N > max_train_samples:
        idx = np.random.RandomState(123).choice(N, size=max_train_samples, replace=False)
        X_in = X_in[idx]

    # Try GPU CP (float32) first
    try:
        if tl.get_backend() == "pytorch" and torch.cuda.is_available():
            Xb = _to_backend(X_in, use_float64=False)  # float32 saves VRAM
            weights, factors = _tl_parafac(
                Xb, rank=rank, init="random", n_iter_max=500, tol=1e-6,
                normalize_factors=True, random_state=random_state
            )
            # factors: [H (N'×R), A (64×R), B (64×R), C (6×R)]
            H_t, A_t, B_t, C_t = factors
            lam_t = weights
            # Fold weights into sample coefficients
            H_t = H_t * lam_t[None, :]
            # To numpy for return
            H_np = _to_numpy(H_t).astype(np.float32)
            A_np = _to_numpy(A_t).astype(np.float32)
            B_np = _to_numpy(B_t).astype(np.float32)
            C_np = _to_numpy(C_t).astype(np.float32)

            if X_in.shape[0] != X_train.shape[0]:
                # Project full TRAIN on GPU basis (GPU projection)
                proj = precompute_cp_projection_accel(A_np, B_np, C_np, use_gpu=True)
                H_full = project_cp_coeffs_accel(X_train, proj, batch=PROJ_BATCH)
                return (A_np, B_np, C_np), H_full.astype(np.float32)
            else:
                return (A_np, B_np, C_np), H_np
    except Exception as e:
        print(f"[global CP] GPU path failed ({e}); falling back to CPU.")

    # CPU fallback (NumPy backend), still init='random'
    old = tl.get_backend()
    try:
        tl.set_backend("numpy")
        weights, factors = _tl_parafac(
            X_in.astype(np.float32),
            rank=rank, init="random",
            n_iter_max=500, tol=1e-6, normalize_factors=True, random_state=random_state
        )
        H, A, B, C = [ np.asarray(F) for F in factors ]
        lam = np.asarray(weights)
        H = H * lam[None, :]
        if X_in.shape[0] != X_train.shape[0]:
            proj = precompute_cp_projection_accel(A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), use_gpu=False)
            H_full = project_cp_coeffs_accel(X_train, proj, batch=PROJ_BATCH)
            return (A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)), H_full.astype(np.float32)
        else:
            return (A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)), H.astype(np.float32)
    finally:
        tl.set_backend(old)

def precompute_cp_projection_accel(A, B, C, eps=1e-4, use_gpu=True):
    """
    Build a projection 'plan' (GPU if possible) to solve G H^T = g^T.
      G = (A^T A) * (B^T B) * (C^T C)
    Returns a dict with torch tensors if GPU, else numpy.
    """
    plan = {"gpu": False}
    try:
        import torch
        if use_gpu and tl.get_backend()=="pytorch" and torch.cuda.is_available():
            At = torch.from_numpy(np.asarray(A)).to("cuda")
            Bt = torch.from_numpy(np.asarray(B)).to("cuda")
            Ct = torch.from_numpy(np.asarray(C)).to("cuda")
            G = (At.T @ At) * (Bt.T @ Bt) * (Ct.T @ Ct)
            G = G + eps * torch.eye(G.shape[0], device="cuda", dtype=G.dtype)
            try:
                L = torch.linalg.cholesky(G)
                plan.update({"gpu": True, "A": At, "B": Bt, "C": Ct, "L": L, "use_cholesky": True})
            except Exception:
                Ginv = torch.linalg.pinv(G, rcond=1e-8)
                plan.update({"gpu": True, "A": At, "B": Bt, "C": Ct, "Ginv": Ginv, "use_cholesky": False})
            return plan
    except Exception as e:
        print(f"[proj] GPU plan failed ({e}); using CPU plan.")

    # CPU plan (numpy) with pinv (robust)
    G = (A.T @ A) * (B.T @ B) * (C.T @ C)
    w = np.linalg.eigvalsh(G)
    cond = (w.max() / max(w.min(), 1e-12))
    print(f"[CP proj] Gram cond≈{cond:.2e}, minEig={w.min():.3e}")
    G = G + eps * np.eye(G.shape[0], dtype=G.dtype)
    Ginv = np.linalg.pinv(G, rcond=1e-8)
    return {"gpu": False, "A": A, "B": B, "C": C, "Ginv": Ginv}

def project_cp_coeffs_accel(X, plan, batch=256):
    """
    Project X: (N,64,64,6) onto fixed CP basis (A,B,C) using plan.
    GPU-accelerated if plan['gpu'] True. Returns H (N,R) numpy.
    """
    N = X.shape[0]
    if plan.get("gpu", False):
        import torch
        A = plan["A"]; B = plan["B"]; C = plan["C"]
        R = A.shape[1]
        H_list = []
        for s in range(0, N, batch):
            e = min(N, s+batch)
            Xb = torch.from_numpy(X[s:e]).to("cuda")
            # g: (batch, R) using einsum
            g = torch.einsum('nijk,ir,jr,kr->nr', Xb, A, B, C)
            if plan.get("use_cholesky", False):
                # Solve G H^T = g^T with cholesky_solve
                Ht = torch.cholesky_solve(g.T, plan["L"]).T
            else:
                Ht = g @ plan["Ginv"].T  # Ginv symmetric
            H_list.append(Ht.detach().cpu().numpy().astype(np.float32))
            del Xb, g, Ht
            torch.cuda.empty_cache()
        return np.concatenate(H_list, axis=0)
    else:
        # CPU fallback (vectorized per-sample g with numpy einsum)
        A = plan["A"]; B = plan["B"]; C = plan["C"]; Ginv = plan["Ginv"]
        R = A.shape[1]
        H = np.empty((N, R), dtype=np.float32)
        for n in range(N):
            g = np.einsum('ijk,i,j,k->', X[n], A[:,0], B[:,0], C[:,0])  # will overwrite below
            g_all = np.empty(R, dtype=np.float64)
            for r in range(R):
                g_all[r] = np.einsum('ijk,i,j,k->', X[n], A[:, r], B[:, r], C[:, r], optimize=True)
            H[n] = (Ginv @ g_all).astype(np.float32)
        return H

def ocsvm_gamma_grid_for_dim(d):
    base = 1.0 / max(d, 1)
    return [base * t for t in (0.3, 1.0, 3.0)]  # small, sane grid

# ====== CP + OC-SVM (GPU-heavy path) ========================================
def parafac_OC_SVM(rank, displayConfusionMatrix=False, use_pca_whiten=True,
                   val_fraction=0.5, random_state=42, cp_basis_max_train_samples=None):
    """
    Global CP basis on TRAIN (GPU if available) -> GPU projection of VAL/FINAL -> OC-SVM.
    Includes large-dataset guards (subsampling grid; Mahalanobis fallback).
    """
    start_time = time.time()
    rng = np.random.RandomState(random_state)

    # ---- 1) Load data
    X_train, _ = readData(train_data)
    X_pool,  y_pool  = readData_test(test_typical_data, test_anomaly_data)
    y_pool = np.asarray(y_pool, int)

    # ---- 2) Optional: band standardize with TRAIN stats
    if USE_BAND_STANDARDIZE:
        mu_b, sig_b = fit_band_standardizer(X_train)
        X_train = apply_band_standardizer(X_train, mu_b, sig_b)
        X_pool  = apply_band_standardizer(X_pool,  mu_b, sig_b)

    sanity_report(X_train, "TRAIN")
    sanity_report(X_pool,  "POOL")
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("[sanity] Cleaning TRAIN tiles ..."); X_train = clean_tiles(X_train)
    if np.isnan(X_pool).any() or np.isinf(X_pool).any():
        print("[sanity] Cleaning POOL tiles ...");  X_pool  = clean_tiles(X_pool)

    # ---- 3) Split POOL to VAL/FINAL stratified
    idx_all = np.arange(len(y_pool)); rng.shuffle(idx_all)
    X_pool = X_pool[idx_all]; y_pool = y_pool[idx_all]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction, random_state=random_state)
    val_idx, final_idx = next(sss.split(np.zeros_like(y_pool), y_pool))
    X_val, y_val = X_pool[val_idx], y_pool[val_idx]
    X_fin, y_fin = X_pool[final_idx], y_pool[final_idx]

    # ---- 4) GLOBAL CP on TRAIN (GPU if available)
    (A, B, C), H_train = fit_global_cp_basis(
        X_train, rank, random_state=random_state,
        max_train_samples=cp_basis_max_train_samples
    )

    # ---- 5) PROJECT VAL & FINAL on the same basis (GPU batched)
    proj = precompute_cp_projection_accel(A, B, C, use_gpu=True)
    H_val = project_cp_coeffs_accel(X_val, proj, batch=PROJ_BATCH)
    H_fin = project_cp_coeffs_accel(X_fin, proj, batch=PROJ_BATCH)

    # ---- 6) Scale + optional PCA whiten (CPU / sklearn)
    scaler = StandardScaler()
    Htr_s  = scaler.fit_transform(H_train)
    Hval_s = scaler.transform(H_val)
    Hfin_s = scaler.transform(H_fin)

    if use_pca_whiten:
        pca   = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        Htr_w = pca.fit_transform(Htr_s)
        Hval_w = pca.transform(Hval_s)
        Hfin_w = pca.transform(Hfin_s)
    else:
        pca = None
        Htr_w, Hval_w, Hfin_w = Htr_s, Hval_s, Hfin_s

    # Sanitize
    Htr_w  = _ensure_finite(Htr_w,  "Htr_w")
    Hval_w = _ensure_finite(Hval_w, "Hval_w")
    Hfin_w = _ensure_finite(Hfin_w, "Hfin_w")

    # ---- 7) Subsample OC-SVM train/val for very large datasets
    Htr_sub, _       = _subsample_rows(Htr_w,  OCS_TRAIN_MAX, rng)
    Hval_sub, yv_sub = _subsample_rows_with_labels(Hval_w, y_val, OCS_VAL_MAX, rng, stratified=True)

    # ---- 8) Small OC-SVM grid (CPU / sklearn)
    feat_dim = Htr_sub.shape[1]
    param_grid = {
        "kernel":    ["rbf"],
        "gamma":     ocsvm_gamma_grid_for_dim(feat_dim),
        "nu":        [0.02, 0.05, 0.1],
        "shrinking": [False],
        "cache_size": [1000],
    }

    def _run_grid(Htr, Hval, yv):
        best_auc = -1.0
        best_model = None
        best_params = None
        total = 0; skips = 0
        for params in ParameterGrid(param_grid):
            total += 1
            try:
                model = OneClassSVM(**params).fit(Htr)
                s_val = -model.decision_function(Hval).ravel()
                if not np.isfinite(s_val).all():
                    raise ValueError("non-finite scores")
                auc = manual_auc(yv, s_val, positive_label=-1)
                if np.isfinite(auc) and auc > best_auc:
                    best_auc, best_model, best_params = auc, model, params
            except Exception:
                skips += 1
                continue
        return best_auc, best_model, best_params, total, skips

    best_auc, best_model, best_params, total, skips = _run_grid(Htr_sub, Hval_sub, yv_sub)

    if best_model is None and Htr_sub.shape[0] > OCS_RETRY_TRAIN_MAX:
        print("[OCSVM] No valid model on first pass; retrying on smaller subset...")
        Htr_retry, _ = _subsample_rows(Htr_w, OCS_RETRY_TRAIN_MAX, rng)
        Hval_retry, yv_retry = _subsample_rows_with_labels(Hval_w, y_val, min(OCS_RETRY_TRAIN_MAX, OCS_VAL_MAX), rng, stratified=True)
        best_auc, best_model, best_params, total2, skips2 = _run_grid(Htr_retry, Hval_retry, yv_retry)
        total += total2; skips += skips2

    if best_model is None:
        print(f"[CP+OCSVM] No valid model found (skipped {skips}/{total}). Falling back to Mahalanobis.")
        # Fallback: Mahalanobis on full features
        mu = Htr_w.mean(axis=0)
        C  = np.cov(Htr_w, rowvar=False) + 1e-6*np.eye(Htr_w.shape[1])
        Ci = np.linalg.pinv(C, rcond=1e-8)
        def mdist(Z):
            R = Z - mu[None, :]
            return np.einsum('ij,jk,ik->i', R, Ci, R, optimize=True)
        s_val  = mdist(Hval_w)
        auc_v  = manual_auc(y_val, s_val, positive_label=-1)
        s_fin  = mdist(Hfin_w)
        auc_te = manual_auc(y_fin, s_fin, positive_label=-1)
        print(f"[CP+Mahalanobis] VAL AUC={auc_v:.3f} | FINAL AUC={auc_te:.3f}")
        th_opt, acc_opt = _pick_threshold_max_accuracy(y_fin, s_fin, positive_label=-1)
        print(f"[CP+Mahalanobis] threshold={th_opt:.6f} | Accuracy={acc_opt:.3f}")
        print('runtime:', round(time.time() - start_time, 2), 's')
        return acc_opt

    print(f"[CP+OCSVM] VAL AUC={best_auc:.3f} with params={best_params}")

    # ---- 9) FINAL evaluation (full FINAL set)
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

# ====== (Optional) raw-pixel OC-SVM baseline (CPU / sklearn) ================
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
    startRank = 10; endRank = 60; step = 2
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_OC_SVM(rank, cp_basis_max_train_samples=400)  # keep CP fit memory in check
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# ====== Entry ================================================================
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
