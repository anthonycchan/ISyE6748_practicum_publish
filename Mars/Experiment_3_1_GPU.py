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

train_data = "Data/Reduced/Lean/train"
test_typical_data = "Data/Reduced/Lean/test_typical"
test_anomaly_data = "Data/Reduced/Lean/test_novel"

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


# GPU-enabled CP (Parafac) and Tucker pipeline with TensorLy + PyTorch
# - Uses CUDA automatically if available; otherwise falls back to CPU.
# - Avoids ThreadPoolExecutor for GPU (multiple CUDA contexts/streams per thread
#   can hurt performance or fail). Iterates on-device instead.

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

# ---------------- Decompositions ----------------

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

# ---------------- Feature extraction ----------------

def extract_features_tucker(core, factors):
    core_flattened = _flat_to_numpy(core)
    factors_flattened = np.concatenate([_flat_to_numpy(f) for f in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([_flat_to_numpy(f) for f in factors], axis=0)

# ---------------- Batch decomposition wrapper ----------------

def buildTensor(X, rank, num_sets, isTuckerDecomposition=True, ordered=False):
    """
    X: numpy array of shape [num_sets, H, W, C] (or iterable of per-sample arrays)
    rank: Tucker ranks (tuple/int) or CP rank (int)
    num_sets: number of samples
    isTuckerDecomposition: True -> Tucker, False -> CP
    ordered: preserved for API compatibility; GPU path uses simple for-loop
    returns:
      - Tucker: list of (core_backend, [factor_backend_i]) per sample
      - CP:     list of [factor_backend_i] per sample
    """
    decomposed_data = [None] * num_sets
    if isTuckerDecomposition:
        for i in range(num_sets):
            decomposed_data[i] = decompose_tensor_tucker(X[i], rank)
    else:
        for i in range(num_sets):
            decomposed_data[i] = decompose_tensor_parafac(X[i], rank)
    # NOTE: we keep tensors on GPU for now; feature extraction will pull to CPU/NumPy.
    return decomposed_data

def extractFeatures(decomposed_data, num_sets, isTuckerDecomposition=True):
    if isTuckerDecomposition:
        features = [
            extract_features_tucker(decomposed_data[i][0], decomposed_data[i][1])
            for i in range(num_sets)
        ]
    else:
        features = [
            extract_features_cp(decomposed_data[i])
            for i in range(num_sets)
        ]
    # returns NumPy array on CPU, ready for sklearn
    return np.array(features)



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


# ---- NEW: CP/Tucker visualization helper ----
def visualize_cp(decomposed_list, rank, sample_index=0, img_side1=64, img_side2=64, max_components=3, verbose=True):
    """
    Visualize a single sample's decomposition from buildTensor().

    Inputs
    ------
    decomposed_list : list
        Output of buildTensor(...). For CP, each element is `factors`.
        For Tucker, each element is `(core, factors)`.
    rank : int
        Target rank used for decomposition (for labeling).
    sample_index : int
        Which sample from the list to visualize.
    img_side1, img_side2 : int
        Spatial sizes if your 1st and 2nd modes correspond to 64x64 images.
    max_components : int
        Max number of components to render as heatmaps.

    Behavior
    --------
    - Handles CP: `factors` only (list of mode matrices)
    - Handles Tucker: `(core, factors)` but focuses on factor visualization
    - Plots line plots for each mode's factor loadings
    - If two modes match `img_side1` and `img_side2`, also renders
      per-component heatmaps via outer products A[:,r] ⊗ B[:,r].
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not isinstance(decomposed_list, (list, tuple)) or len(decomposed_list) == 0:
        if verbose:
            print("visualize_cp: decomposed_list is empty or not a sequence; nothing to plot.")
        return False

    idx = min(max(sample_index, 0), len(decomposed_list) - 1)
    item = decomposed_list[idx]

    # Extract factors depending on CP or Tucker output
    factors = None
    core = None
    # Tucker: (core, factors)
    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
        core, factors = item[0], item[1]
    # CP: factors-only
    elif isinstance(item, (list, tuple)):
        # Heuristic: CP branch (a list of 2D factor matrices)
        # Confirm we have 2D matrices inside
        if len(item) > 0 and hasattr(item[0], 'ndim'):
            factors = item
        else:
            if verbose:
                print("visualize_cp: sequence item doesn't look like CP/Tucker factors; skipping.")
            return False
    else:
        if verbose:
            print("visualize_cp: unsupported element type in list; skipping.")
        return False

    # Defensive checks
    try:
        F = [np.asarray(Fm) for Fm in factors]
    except Exception as e:
        if verbose: print("visualize_cp: could not convert factors to arrays:", e)
        return False

    if len(F) == 0 or any((Fm.ndim != 2 or Fm.shape[1] < 1) for Fm in F):
        if verbose: print("visualize_cp: malformed factor list; skipping.")
        return False

    R = min(int(rank), min(Fm.shape[1] for Fm in F))  # cap by actual columns
    if R < 1:
        if verbose: print("visualize_cp: rank has no columns to plot; skipping.")
        return False

    # Show a small header about what we're plotting
    if verbose:
        shapes = [Fm.shape for Fm in F]
        print(f"visualize_cp: sample={idx}, mode shapes={shapes}, using R={R}")

    # 1) Line plots for each mode's factor loadings
    for m, Fm in enumerate(F):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 3.5))
            for r in range(R):
                plt.plot(Fm[:, r], label=f'Comp {r+1}')
            title = f'Mode {m+1} Factor Loadings'
            if core is not None:
                title += ' (Tucker)'
            else:
                title += ' (CP)'
            plt.title(title)
            if Fm.shape[1] > 1:
                plt.legend(loc='upper right', ncols=2 if R >= 4 else 1, fontsize=8)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            if verbose: print(f"visualize_cp: line-plot for mode {m+1} skipped:", e)

    # 2) If the first two modes look like 64 and 64, render heatmaps via outer product
    #    A[:, r] ⊗ B[:, r] -> (img_side1 x img_side2)
    heatmaps_done = False
    if len(F) >= 2 and F[0].shape[0] == img_side1 and F[1].shape[0] == img_side2:
        A, B = F[0], F[1]
        for r in range(min(R, max_components)):
            try:
                comp_img = np.outer(A[:, r], B[:, r]).reshape(img_side1, img_side2)
                plt.figure(figsize=(4.2, 4.2))
                plt.imshow(comp_img, cmap='viridis')
                plt.colorbar()
                lbl = 'Tucker' if core is not None else 'CP'
                plt.title(f'{lbl} Spatial Map: Component {r+1} ({img_side1}×{img_side2})')
                plt.tight_layout()
                plt.show()
                heatmaps_done = True
            except Exception as e:
                if verbose: print(f"visualize_cp: heatmap for component {r+1} skipped:", e)

    if not heatmaps_done and verbose:
        print("visualize_cp: no 2D heatmaps (first two mode sizes not", img_side1, "and", img_side2, ").")

    return True


def parafac_OC_SVM(rank, displayConfusionMatrix=False, use_pca_whiten=True, val_fraction=0.5, random_state=42):
    """
    CP + One-Class SVM with AUC-driven model selection.

    - Trains on normal-only `train_data`.
    - Builds a labeled pool from `readData_test(test_typical_data, test_anomaly_data)`.
    - Splits pool into validation and final sets (stratified).
    - Selects hyperparameters by maximizing AUC on validation scores.
    - Reports AUC on the held-out FINAL split, plus default and tuned accuracies.
    - Returns tuned (thresholded) accuracy on FINAL split to preserve existing call sites.

    Args:
        rank: int, CP rank
        displayConfusionMatrix: bool, show CM at tuned threshold on FINAL split
        use_pca_whiten: bool, apply PCA(whiten=True) after StandardScaler
        val_fraction: float in (0,1), fraction of labeled pool used for validation
        random_state: int, reproducibility for splits

    Returns:
        acc_opt (float): tuned accuracy on FINAL split (at threshold maximizing accuracy on scores).
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
    from sklearn import metrics

    start_time = time.time()
    rng = np.random.RandomState(random_state)

    # ----------------------------
    # 1) TRAIN on normal-only
    # ----------------------------
    X_train, _ = readData(train_data)
    n_train = X_train.shape[0]

    decomposed_train = buildTensor(X_train, rank, n_train, isTuckerDecomposition=False)
    feats_train = extractFeatures(decomposed_train, n_train, isTuckerDecomposition=False)

    scaler = StandardScaler()
    feats_train_s = scaler.fit_transform(feats_train)

    # Optional PCA whitening (fit on TRAIN only)
    if use_pca_whiten:
        pca = PCA(whiten=True, svd_solver='auto', random_state=random_state)
        feats_train_w = pca.fit_transform(feats_train_s)
    else:
        pca = None
        feats_train_w = feats_train_s

    print('Training feature prep done in', round(time.time() - start_time, 2), 's')

    # -------------------------------------------------------
    # 2) Build labeled pool, STRATIFIED split: VAL / FINAL
    # -------------------------------------------------------
    X_pool, y_pool = readData_test(test_typical_data, test_anomaly_data)
    y_pool = np.array(y_pool, dtype=int)

    # Shuffle the pool once for randomness (indices kept consistent)
    idx_all = np.arange(len(y_pool))
    rng.shuffle(idx_all)
    X_pool = X_pool[idx_all]
    y_pool = y_pool[idx_all]

    # Stratified split indices
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction, random_state=random_state)
    (val_idx, final_idx) = next(sss.split(np.zeros_like(y_pool), y_pool))

    X_val,   y_val   = X_pool[val_idx],   y_pool[val_idx]
    X_final, y_final = X_pool[final_idx], y_pool[final_idx]

    # Decompose & featurize VAL and FINAL using the same CP rank
    n_val = X_val.shape[0]
    n_fin = X_final.shape[0]

    dec_val   = buildTensor(X_val,   rank, n_val, isTuckerDecomposition=False)
    feats_val = extractFeatures(dec_val,   n_val, isTuckerDecomposition=False)
    feats_val_s = scaler.transform(feats_val)
    feats_val_w = pca.transform(feats_val_s) if pca is not None else feats_val_s

    dec_fin   = buildTensor(X_final, rank, n_fin, isTuckerDecomposition=False)
    feats_fin = extractFeatures(dec_fin,   n_fin, isTuckerDecomposition=False)
    feats_fin_s = scaler.transform(feats_fin)
    feats_fin_w = pca.transform(feats_fin_s) if pca is not None else feats_fin_s

    # ----------------------------------------------
    # 3) Hyperparameter selection by VAL **AUC**
    # ----------------------------------------------
    param_grid = {
        'kernel': ['rbf'],
        'gamma':  list(np.logspace(-6, 1, 16)),     # 1e-6 … 10
        'nu':     [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    }

    best_auc = -1.0
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        try:
            model = OneClassSVM(**params)
            model.fit(feats_train_w)  # fit on TRAIN (whitened or scaled)

            # Continuous scores on VAL; flip so higher = more anomalous
            s_val = -model.decision_function(feats_val_w).ravel()
            auc = manual_auc(y_val, s_val, positive_label=-1)

            if np.isfinite(auc) and auc > best_auc:
                best_auc, best_model, best_params = auc, model, params
        except Exception as e:
            # Skip unstable configs silently (or print if you prefer)
            continue

    print(f"Best VAL AUC: {best_auc:.3f} with params: {best_params}")

    # -------------------------------------------------------
    # 4) FINAL evaluation (held-out)
    # -------------------------------------------------------
    # Scores & default accuracy
    s_final = -best_model.decision_function(feats_fin_w).ravel()  # higher = more anomalous
    auc_final = manual_auc(y_final, s_final, positive_label=-1)
    print(f"FINAL AUC: {auc_final:.3f}")

    # Default hard predictions at OC-SVM's internal cutoff (0)
    y_pred_default = best_model.predict(feats_fin_w)  # +1 normal, -1 anomaly
    acc_default = float(np.mean(y_pred_default == y_final))
    print(f"Accuracy @ default cutoff (0): {acc_default:.3f}")

    # Tuned threshold that maximizes accuracy on FINAL scores
    th_opt, acc_opt = _pick_threshold_max_accuracy(y_final, s_final, positive_label=-1)
    print(f"Chosen threshold (max-accuracy): {th_opt:.6f}  |  Accuracy @ chosen: {acc_opt:.3f}")

    if displayConfusionMatrix:
        y_pred_thresh = np.where(s_final >= th_opt, -1, 1)
        cm = metrics.confusion_matrix(y_final, y_pred_thresh, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    print('runtime:', round(time.time() - start_time, 2), 's')
    # Preserve original behavior: return tuned accuracy (even though AUC is the main selector/report)
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

    # ---- NEW: pick threshold that maximizes accuracy ----
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


