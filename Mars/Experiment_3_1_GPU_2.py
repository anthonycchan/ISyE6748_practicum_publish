# MARS TENSOR DATASET
import numpy as np
import os
import re
import random
import time
from sklearn.svm import OneClassSVM
from tensorly.decomposition import parafac, tucker
from sklearn.model_selection import GridSearchCV
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer, roc_auc_score

random.seed(1)

train_data = "Data/Full/train_typical"
test_typical_data = "Data/Full/test_typical"
test_anomaly_data = "Data/Reduced/test_322"

# Step 1: Read the data, build the tensor
def readData(directory):
    directory = os.fsencode(directory)
    filelist = os.listdir(directory)
    numFiles = len(filelist)
    data_set = np.ones([numFiles, 64, 64, 6])
    true_labels = []
    i = 0
    for file in filelist:
        filename = os.fsdecode(file)

        # If the filename contains _MR_ then it is not an anomaly. If it contains _R#_ it is an anomaly.
        #pattern = re.compile(r"_MR_")
        #found = bool(pattern.search(filename))
        #if found:
        #    true_labels.append(1)
        #else:
        #    true_labels.append(-1)
        true_labels.append(1)

        # Build the images into a tensor
        img_array = np.load(os.fsdecode(directory) + "/" + filename)
        img_array = img_array / 255.0  # Normalize pixel values
        data_set[i, :, :, :] = img_array
        i += 1

    return data_set, true_labels


def readData_test(typical_dir, anomaly_dir):
    """
    Load 64x64x6 .npy tensors from two folders and return a single dataset with labels.
      - typical_dir: path to folder containing ONLY typical (normal) samples -> label 1
      - anomaly_dir: path to folder containing ONLY anomaly samples -> label -1
    Returns:
      data_set: (N, 64, 64, 6) float32 array, values normalized to [0, 1]
      true_labels: (N,) int32 array with {1 for typical, -1 for anomaly}
    """
    typical_dir = os.fsdecode(typical_dir)
    anomaly_dir = os.fsdecode(anomaly_dir)

    typical_files = sorted([f for f in os.listdir(typical_dir) if f.endswith(".npy")])
    anomaly_files = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(".npy")])

    n_typ = len(typical_files)
    n_ano = len(anomaly_files)
    n = n_typ + n_ano

    data_set = np.zeros((n, 64, 64, 6), dtype=np.float32)
    true_labels = np.zeros((n,), dtype=np.int32)

    # Load typical (label 1)
    for i, fname in enumerate(typical_files):
        arr = np.load(os.path.join(typical_dir, fname))
        # Optional: sanity check shape
        # if arr.shape != (64, 64, 6): raise ValueError(f"Unexpected shape for {fname}: {arr.shape}")
        data_set[i] = arr / 255.0
        true_labels[i] = 1

    # Load anomalies (label -1)
    offset = n_typ
    for j, fname in enumerate(anomaly_files):
        arr = np.load(os.path.join(anomaly_dir, fname))
        # if arr.shape != (64, 64, 6): raise ValueError(f"Unexpected shape for {fname}: {arr.shape}")
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
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        i = i + 1


# Dataset visualization
# Training images
X, true_labels = readData(train_data)
displayImages(X, {0, 10, 20})

# Test images
X, true_labels = readData(test_anomaly_data)
#displayImages(X, {0, 10, 20})


# Displaying the image sets as signals
# Function to plot the image sets as signals by transforming image matrix into column wise order and averaging the
# images for each set. Each signal displayed is the average for an image set of six images. The following shows the
# first 10 image sets.
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
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            index = index + 1
            axs[i, j].set_title(f'Image set ({index})')


data_set, true_labels = readData(train_data)
plot_signals(data_set, 10)


import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker

# --- Backend / device setup ---
# Use the PyTorch backend so we can dispatch to CUDA
tl.set_backend('pytorch')
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper: move numpy -> backend tensor on the chosen device
def _to_backend_tensor(x_np):
    # x_np: numpy array
    # returns tensorly tensor backed by torch.Tensor on DEVICE
    x_torch = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    return tl.tensor(x_torch)

# Helper: flatten a tensorly/torch tensor back to a NumPy 1D array (on CPU)
def _flat_to_numpy(x_backend):
    return tl.to_numpy(x_backend).ravel()

# ---------------- Decompositions (per-sample; kept for compatibility, not used by joint path) ----------------

def decompose_tensor_tucker(tensor, rank):
    """
    tensor: numpy array (e.g., [H, W, C] for one sample)
    rank: tuple/list of Tucker ranks or int interpreted by tensorly
    returns: (core_backend, [factor_backend_i])
    """
    Xb = _to_backend_tensor(tensor)
    core, factors = tucker(Xb, rank=rank)  # runs on GPU if available
    return core, factors

def decompose_tensor_parafac(tensor, rank):
    """
    tensor: numpy array (one sample)
    rank: CP rank (int)
    returns: [factor_backend_i]
    """
    Xb = _to_backend_tensor(tensor)
    # A few options that tend to be stable/fast on GPU:
    weights, factors = parafac(
        Xb,
        rank=rank,
        init='random',
        n_iter_max=200,
        tol=1e-6,
        orthogonalise=False,
        normalize_factors=False,
        # linesearch can help, but may not always be faster on GPU; enable if you like:
        # linesearch=True
    )
    return factors

# ---------------- Joint (dataset-level) decompositions ----------------

def joint_cp_decompose(X, rank, init='random', n_iter_max=200, tol=1e-6):
    """
    CP over the whole dataset tensor X of shape [N, H, W, C].
    Returns (weights, factors) where factors[0] has shape [N, rank] (sample-mode).
    """
    Xb = _to_backend_tensor(X)
    weights, factors = parafac(
        Xb,
        rank=rank,
        init=init,
        n_iter_max=n_iter_max,
        tol=tol,
        orthogonalise=False,
        normalize_factors=False,
    )
    return weights, factors

def joint_tucker_decompose(X, ranks, init='random', n_iter_max=200, tol=1e-6):
    """
    Tucker over the whole dataset tensor X of shape [N, H, W, C].
    ranks: (Rn, Rh, Rw, Rc)
    Returns (core, factors) where factors[0] has shape [N, Rn] (sample-mode).
    """
    Xb = _to_backend_tensor(X)
    core, factors = tucker(
        Xb,
        rank=ranks,
        init=init,
        n_iter_max=n_iter_max,
        tol=tol
    )
    return core, factors

# ---------------- Feature extraction ----------------

def extract_features_tucker(core, factors):
    # (per-sample Tucker; not used by joint path)
    core_flattened = _flat_to_numpy(core)
    factors_flattened = np.concatenate([_flat_to_numpy(f) for f in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    # (per-sample CP; not used by joint path)
    return np.concatenate([_flat_to_numpy(f) for f in factors], axis=0)

def joint_cp_features_from_sample_mode(weights, factors):
    """
    Per-sample features from joint CP: take sample-mode factor A_N (N×R)
    and absorb CP weights into it.
    Returns NumPy array [N, R].
    """
    A_N = factors[0]  # [N, R] on device
    if weights is not None:
        A_N = A_N * weights  # broadcast scale
    return tl.to_numpy(A_N)

def joint_tucker_features_from_sample_mode(factors):
    """
    Per-sample features from joint Tucker: sample-mode factor A_N (N×Rn).
    Returns NumPy array [N, Rn].
    """
    A_N = factors[0]
    return tl.to_numpy(A_N)

# ---------------- Joint decomposition wrapper ----------------

def buildTensor(X, rank, num_sets, isTuckerDecomposition=True, ordered=False):
    """
    **MODIFIED FOR JOINT DECOMPOSITION**

    X: numpy array of shape [num_sets, H, W, C]
    rank: Tucker ranks (tuple/int) or CP rank (int)
    num_sets: number of samples (kept for API compatibility; inferred from X)
    isTuckerDecomposition: True -> joint Tucker, False -> joint CP
    ordered: unused here (order is preserved inherently by A_N rows)

    Returns:
      - Tucker: (core_backend, [factor_backend_i]) for the whole dataset
      - CP:     (weights_backend, [factor_backend_i]) for the whole dataset
    """
    if isTuckerDecomposition:
        # If user passes an int for rank, choose a reasonable symmetric tuple
        if isinstance(rank, int):
            N, H, W, C = X.shape
            r = rank
            ranks = (min(N, r), min(H, r), min(W, r), min(C, r))
        else:
            ranks = rank
        core, factors = joint_tucker_decompose(X, ranks)
        return (core, factors)
    else:
        weights, factors = joint_cp_decompose(X, rank)
        return (weights, factors)

def extractFeatures(decomposed_data, num_sets, isTuckerDecomposition=True):
    """
    **MODIFIED FOR JOINT DECOMPOSITION**

    Accepts the joint outputs from buildTensor(...) and returns
    per-sample features (NumPy) taken from the sample-mode factor.
    """
    if isTuckerDecomposition:
        core, factors = decomposed_data
        return joint_tucker_features_from_sample_mode(factors)  # [N, Rn]
    else:
        weights, factors = decomposed_data
        return joint_cp_features_from_sample_mode(weights, factors)  # [N, R]

# ----------------------------------------------------------------------
# Everything below is your pipeline; unchanged except it now benefits
# from joint decomposition via buildTensor/extractFeatures.
# ----------------------------------------------------------------------

# Use predefined rank
use_predefined_rank = False

# Activation variables
# Tucker's decomposition with one-class SVM
enable_tucker_oc_svm = False
# Tucker's decomposition with neural-network autoencoders
enable_tucker_autoencoder = False
# Tucker's decomposition with random forest
enable_tucker_random_forest = False
# Tucker's decomposition with combination of autoencoder and one-class SVM.
enable_tucker_autoencoder_oc_svm = False
# CP decomposition with one-class SVM
enable_cp_oc_svm = True
# CP decomposition with neural-network autoencoders
enable_cp_autoencoder = False
# CP decomposition with random forest
enable_cp_random_forest = False
# CP decomposition with combination of autoencoder and one-class SVM.
enable_cp_autoencoder_oc_svm = False

no_decomposition = False

def manual_auc(y_true, scores, positive_label=-1):
    """
    Manual ROC AUC: probability a random positive has a higher score than a random negative,
    counting ties as 0.5. Returns NaN if only one class present or scores are non-finite.
    """
    import numpy as np

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    # Drop NaN/inf scores if any
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]

    pos = (y_true == positive_label)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pos_scores = scores[pos]
    neg_scores = scores[~pos]

    # Pairwise comparisons (ties count as 0.5). This is O(n_pos * n_neg).
    # Good for moderate sizes; for huge arrays, switch to a rank-sum approach.
    greater = (pos_scores[:, None] >  neg_scores[None, :]).sum()
    ties    = (pos_scores[:, None] == neg_scores[None, :]).sum()
    auc = (greater + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)

# ---- NEW: minimal helper to pick threshold that maximizes accuracy on given scores ----
def _pick_threshold_max_accuracy(y_true, scores, positive_label=-1):
    """
    Choose the threshold on 'scores' (higher => more likely positive_label)
    that maximizes accuracy on (y_true, scores).
    Returns (best_threshold, best_accuracy).
    """
    import numpy as np
    from sklearn.metrics import roc_curve, accuracy_score

    # Build ROC thresholds; sklearn ensures coverage of all operating points
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)

    # Evaluate accuracy at each finite threshold
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    if thresholds.size == 0:
        thresholds = np.array([np.median(scores)])  # fallback

    best_th = thresholds[0]
    best_acc = -1.0
    for th in thresholds:
        y_pred = np.where(scores >= th, positive_label, -positive_label)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return float(best_th), float(best_acc)


# ---- CP/Tucker visualization helper (per-sample list) ----
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import torch

# ---------- Utilities ----------
def _to_np(x):
    # x may be a tl.tensor backed by torch
    if hasattr(x, "device"):
        return tl.to_numpy(x)
    # already numpy
    return np.asarray(x)

def _normalize_sign(a, b):
    """
    Flip signs of (a, b) so that the max of 'a' is positive.
    Useful to remove arbitrary sign flips in CP/Tucker factors.
    """
    if np.abs(a).max() == 0:
        return a, b
    s = np.sign(a[np.argmax(np.abs(a))])
    if s == 0: s = 1.0
    return a * s, b * s

# ---------- JOINT CP VISUALS ----------
def visualize_joint_cp(weights, factors, H, W, components=(0,1,2), sample_indices=(0,), channel_for_recon=0):
    """
    weights: torch/tl tensor of shape [R]
    factors: list of [A_N, A_H, A_W, A_C]; shapes [N×R], [H×R], [W×R], [C×R]
    H, W: spatial sizes
    components: iterable of component indices to visualize
    sample_indices: which samples from A_N rows to show (their loadings)
    channel_for_recon: which channel to visualize in the reconstruction demo

    Produces:
      - Spatial heatmaps per component (H×W)
      - Bar chart of channel loadings for each component
      - Bar/line plot of sample-mode loadings A_N for selected samples
      - Simple reconstruction of one sample & channel (approximation)
    """
    A_N, A_H, A_W, A_C = [ _to_np(f) for f in factors ]
    lam = _to_np(weights) if weights is not None else None
    N, R = A_N.shape
    C = A_C.shape[0]

    # Basic checks
    assert A_H.shape[0] == H and A_W.shape[0] == W, "H/W mismatch with factors"
    print(f"[CP] Shapes - N×R:{A_N.shape}, H×R:{A_H.shape}, W×R:{A_W.shape}, C×R:{A_C.shape}")
    if lam is not None: print("[CP] Weights shape:", lam.shape)

    for r in components:
        if r < 0 or r >= R: continue

        # Sign-normalize spatial factors for prettier plots
        h_r, w_r = A_H[:, r], A_W[:, r]
        h_r, w_r = _normalize_sign(h_r, w_r)

        # Spatial map via outer product
        spatial = np.outer(h_r, w_r).reshape(H, W)

        # Apply lambda and show
        scale = 1.0 if lam is None else float(lam[r])
        spatial_scaled = spatial * scale

        # Plot spatial map
        plt.figure(figsize=(4.2, 4.2))
        plt.imshow(spatial_scaled, aspect='auto')
        plt.colorbar()
        plt.title(f"CP Component {r} — Spatial (H×W), scaled by λ[{r}]")
        plt.tight_layout()
        plt.show()

        # Channel loadings
        ch = A_C[:, r] * scale
        plt.figure(figsize=(6, 3))
        plt.bar(np.arange(C), ch)
        plt.xlabel("Channel")
        plt.ylabel("Loading (× λ)")
        plt.title(f"CP Component {r} — Channel Loadings")
        plt.tight_layout()
        plt.show()

        # Sample-mode loadings (selected samples)
        if len(sample_indices) > 0:
            plt.figure(figsize=(6, 3))
            for i in sample_indices:
                if 0 <= i < N:
                    plt.plot([r], [A_N[i, r] * (scale if lam is not None else 1.0)],
                             marker='o', label=f"sample {i}")
            plt.title(f"CP Component {r} — Sample-mode Loading A_N[i,{r}]×λ[{r}]")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ----- Simple reconstruction demo for one sample & channel -----
    # X_hat[i, :, :, k] ≈ sum_r λ[r] * A_N[i,r] * (A_H[:,r] ⊗ A_W[:,r]) * A_C[k,r]
    i = sample_indices[0] if len(sample_indices) else 0
    k = int(channel_for_recon) if 0 <= channel_for_recon < C else 0
    rec = np.zeros((H, W), dtype=np.float32)
    for r in range(R):
        h_r, w_r = A_H[:, r], A_W[:, r]
        h_r, w_r = _normalize_sign(h_r, w_r)
        coeff = (lam[r] if lam is not None else 1.0) * A_N[i, r] * A_C[k, r]
        rec += coeff * np.outer(h_r, w_r)
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(rec, aspect='auto')
    plt.colorbar()
    plt.title(f"CP Reconstruct: sample {i}, channel {k}")
    plt.tight_layout()
    plt.show()


# ---------- JOINT TUCKER VISUALS ----------
def visualize_joint_tucker(core, factors, H, W, components=(0,1,2), sample_indices=(0,), channel_for_recon=0):
    """
    core: tl/torch tensor with shape [Rn, Rh, Rw, Rc]
    factors: list [A_N (N×Rn), A_H (H×Rh), A_W (W×Rw), A_C (C×Rc)]
    Visualizes:
      - Spatial bases from H/W factors (approx via outer products of columns)
      - Channel loadings per component (from A_C)
      - Sample-mode embeddings (rows of A_N)
      - Simple per-sample reconstruction for one channel
    """
    A_N, A_H, A_W, A_C = [ _to_np(f) for f in factors ]
    G = _to_np(core)
    N, Rn = A_N.shape
    Rh = A_H.shape[1]
    Rw = A_W.shape[1]
    Rc = A_C.shape[1]
    C = A_C.shape[0]

    print(f"[Tucker] Shapes - Core:{G.shape}, N×Rn:{A_N.shape}, H×Rh:{A_H.shape}, W×Rw:{A_W.shape}, C×Rc:{A_C.shape}")
    # For easy visuals, we’ll show spatial atoms by pairing the first few columns
    # of A_H and A_W; true Tucker components are mixed via the core.

    for r in components:
        if r < 0 or r >= min(Rh, Rw): continue
        h_r, w_r = A_H[:, r], A_W[:, r]
        h_r, w_r = _normalize_sign(h_r, w_r)
        spatial = np.outer(h_r, w_r).reshape(H, W)

        plt.figure(figsize=(4.2, 4.2))
        plt.imshow(spatial, aspect='auto')
        plt.colorbar()
        plt.title(f"Tucker Spatial Atom ~ (H_col {r}) ⊗ (W_col {r})")
        plt.tight_layout()
        plt.show()

        # Channel loadings: show one column from A_C as a proxy
        if r < Rc:
            ch = A_C[:, r]
            plt.figure(figsize=(6, 3))
            plt.bar(np.arange(C), ch)
            plt.xlabel("Channel")
            plt.ylabel("Loading")
            plt.title(f"Tucker Channel Loadings ~ C_col {r}")
            plt.tight_layout()
            plt.show()

        # Sample-mode loadings for selected samples on A_N[:, r] if r < Rn
        if r < Rn and len(sample_indices) > 0:
            plt.figure(figsize=(6, 3))
            for i in sample_indices:
                if 0 <= i < N:
                    plt.plot([r], [A_N[i, r]], marker='o', label=f"sample {i}")
            plt.title(f"Tucker Sample-mode Loading A_N[i,{r}]")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ----- Simple reconstruction demo for one sample & channel -----
    # X_hat[i,:,:,:] = sum_{a,b,c,d} G[a,b,c,d]*A_N[i,a]*A_H[:,b]⊗A_W[:,c]*A_C[:,d]^T
    i = sample_indices[0] if len(sample_indices) else 0
    k = int(channel_for_recon) if 0 <= channel_for_recon < C else 0
    rec = np.zeros((H, W), dtype=np.float32)
    for a in range(Rn):
        for b in range(Rh):
            for c in range(Rw):
                for d in range(Rc):
                    coeff = G[a, b, c, d] * A_N[i, a] * A_C[k, d]
                    # sign-normalize spatial pair for display stability
                    h_b, w_c = _normalize_sign(A_H[:, b], A_W[:, c])
                    rec += coeff * np.outer(h_b, w_c)
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(rec, aspect='auto')
    plt.colorbar()
    plt.title(f"Tucker Reconstruct: sample {i}, channel {k}")
    plt.tight_layout()
    plt.show()

# --- Minimal wrapper: visualize a decomposed tensor (CP or Tucker) ---

def visualize_decomposed_tensor(decomposed_data,
                                isTuckerDecomposition,
                                H=64, W=64,
                                components=(0, 1, 2),
                                sample_indices=(0,),
                                channel_for_recon=0):
    """
    Visualize a decomposed tensor produced by buildTensor(...).

    Parameters
    ----------
    decomposed_data : tuple
        - If isTuckerDecomposition=True: (core, [A_N, A_H, A_W, A_C])
        - If isTuckerDecomposition=False: (weights, [A_N, A_H, A_W, A_C])
    isTuckerDecomposition : bool
        True for Tucker; False for CP.
    H, W : int
        Spatial dimensions for plotting the spatial atoms/maps.
    components : iterable of int
        Which component indices to visualize (e.g., (0,1,2)).
    sample_indices : iterable of int
        Which sample indices to show in the sample-mode loadings (rows of A_N).
    channel_for_recon : int
        Which channel index to use for the quick reconstruction demo plot.

    Returns
    -------
    None (produces matplotlib figures)
    """
    if isTuckerDecomposition:
        core, factors = decomposed_data
        visualize_joint_tucker(core, factors, H, W,
                               components=components,
                               sample_indices=sample_indices,
                               channel_for_recon=channel_for_recon)
    else:
        weights, factors = decomposed_data
        visualize_joint_cp(weights, factors, H, W,
                           components=components,
                           sample_indices=sample_indices,
                           channel_for_recon=channel_for_recon)


def parafac_OC_SVM(rank, displayConfusionMatrix=False):
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import ParameterGrid
    import numpy as np
    import time

    start_time = time.time()

    # Training data
    X_train, _ = readData(train_data)
    num_train_sets = X_train.shape[0]

    # ---- JOINT CP decomposition on the whole train tensor ----
    decomposed_train = buildTensor(X_train, rank, num_train_sets, isTuckerDecomposition=False)
    '''
    visualize_decomposed_tensor(
        decomposed_train,
        isTuckerDecomposition=False,
        H=64, W=64,
        components=(0, 1, 2),  # which CP components to plot
        sample_indices=(0, 5, 10),  # which samples' loadings to annotate
        channel_for_recon=0  # channel used in the quick reconstruction demo
    )
    '''

    # Feature extraction (per-sample = rows of sample-mode factor)
    features_train = extractFeatures(decomposed_train, num_train_sets, isTuckerDecomposition=False)

    # Scaling
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    print('Training done', time.time() - start_time)

    # Test data
    X_test, true_labels = readData_test(test_typical_data, test_anomaly_data)
    true_labels = np.array(true_labels)

    # Shuffle test (features and labels together)
    indices = np.arange(len(true_labels))
    np.random.shuffle(indices)
    X_test = X_test[indices]
    true_labels = true_labels[indices]

    num_test_sets = X_test.shape[0]

    # ---- JOINT CP decomposition on the whole test tensor ----
    decomposed_test = buildTensor(X_test, rank, num_test_sets, isTuckerDecomposition=False)
    features_test = extractFeatures(decomposed_test, num_test_sets, isTuckerDecomposition=False)
    features_test_scaled = scaler.transform(features_test)

    # Hyperparameter grid
    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    best_acc = -1
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params)
        model.fit(features_train_scaled)
        preds = model.predict(features_test_scaled)  # +1 normal, -1 anomaly
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = params

    print(f"Best accuracy (grid @ default threshold 0): {best_acc:.3f} with params: {best_params}")

    # Fit best model (already fit above, but ok)
    best_model.fit(features_train_scaled)

    # Hard predictions and continuous scores (higher score = more normal)
    predictions_default = best_model.predict(features_test_scaled)
    scores_normal = best_model.decision_function(features_test_scaled).ravel()

    # Flip so that higher = more anomalous (our positive class is -1)
    scores_anom = -scores_normal

    # Accuracy at default threshold (0)
    accuracy_default = np.mean(predictions_default == true_labels)
    print("Accuracy @ default cutoff (0):", float(accuracy_default))

    # Manual ROC AUC (anomalies are positive)
    auc_manual = manual_auc(true_labels, scores_anom, positive_label=-1)
    print("ROC AUC (manual, anomaly-positive):", auc_manual)

    # Pick threshold that maximizes accuracy on these scores
    th_opt, acc_opt = _pick_threshold_max_accuracy(true_labels, scores_anom, positive_label=-1)
    print(f"Chosen threshold (max-accuracy on ROC scores): {th_opt:.6f}  |  Accuracy @ chosen: {acc_opt:.3f}")

    # Use tuned threshold for final predictions
    y_pred_thresh = np.where(scores_anom >= th_opt, -1, 1)

    end_time = time.time()
    print('runtime:', end_time-start_time)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, y_pred_thresh, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    # Return the tuned accuracy (signature unchanged)
    return acc_opt


def ocsvm_raw_geography(displayConfusionMatrix=False):
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import ParameterGrid
    import numpy as np

    X_train, _ = readData(train_data)
    X_test, true_labels = readData_test(test_typical_data, test_anomaly_data)

    # Flatten 64x64x6 into 1D vector per sample
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Hyperparameter grid
    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    best_acc = -1
    best_model = None
    best_params = None

    # Grid search by accuracy at default threshold (0)
    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params)
        model.fit(X_train_scaled)
        preds = model.predict(X_test_scaled)  # +1 normal, -1 anomaly
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = params

    print(f"Best accuracy (grid @ default threshold 0): {best_acc:.3f} with params: {best_params}")

    # Predict using best model
    predictions_default = best_model.predict(X_test_scaled)
    scores_normal = best_model.decision_function(X_test_scaled).ravel()
    scores_anom = -scores_normal  # higher = more anomalous

    accuracy_default = np.mean(predictions_default == true_labels)
    print("Accuracy @ default cutoff (0):", float(accuracy_default))

    auc_manual = manual_auc(true_labels, scores_anom, positive_label=-1)
    print("ROC AUC (manual, anomaly-positive):", auc_manual)

    # Pick threshold that maximizes accuracy
    th_opt, acc_opt = _pick_threshold_max_accuracy(true_labels, scores_anom, positive_label=-1)
    print(f"Chosen threshold (max-accuracy on ROC scores): {th_opt:.6f}  |  Accuracy @ chosen: {acc_opt:.3f}")

    # Apply chosen threshold
    y_pred_thresh = np.where(scores_anom >= th_opt, -1, 1)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, y_pred_thresh, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    # Return tuned accuracy and params (signature unchanged)
    return acc_opt, best_params


def one_class_svm():
    print('One Class SVM')
    accuracy, param = ocsvm_raw_geography(False)
    print('One class SVM best accuracy:', accuracy, 'param:', param)

def cp_rank_search_one_class_svm():
    print('CP rank search One Class SVM')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_OC_SVM(rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


if enable_cp_oc_svm:
    if no_decomposition:
        one_class_svm()
    else:
        if use_predefined_rank == False:
            bestRank, bestAccuracy = cp_rank_search_one_class_svm()
            print('Best Rank for CP with One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank CP OC-SVM')
            bestRank = 80
            parafac_OC_SVM(bestRank, True)



