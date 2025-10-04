# MARS TENSOR DATASET — GPU-ENABLED
import os
import re
import random
import warnings

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

# ===============================
# GPU SETUP: TensorLy + PyTorch
# ===============================
import tensorly as tl
tl.set_backend('pytorch')  # use PyTorch backend so decompositions run on GPU via torch

import torch as _torch
_device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')

def _to_torch(x):
    if isinstance(x, np.ndarray):
        return _torch.from_numpy(x.astype(np.float32, copy=False)).to(_device)
    return x.to(_device)

def _to_numpy(x):
    if isinstance(x, _torch.Tensor):
        return x.detach().to('cpu').numpy()
    return np.asarray(x)

# ===============================
# TensorFlow (Keras) GPU behavior
# ===============================
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
except Exception:
    pass

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# Optional: RAPIDS cuML backends
# ===============================
_USE_CUML = False
try:
    from cuml.svm import OneClassSVM as cuOneClassSVM
    from cuml.ensemble import IsolationForest as cuIsolationForest
    _USE_CUML = True
except Exception:
    _USE_CUML = False

def _make_ocsvm(**kwargs):
    if _USE_CUML:
        return cuOneClassSVM(**kwargs)
    else:
        from sklearn.svm import OneClassSVM
        return OneClassSVM(**kwargs)

def _make_isolation_forest(**kwargs):
    if _USE_CUML:
        return cuIsolationForest(**kwargs)
    else:
        from sklearn.ensemble import IsolationForest
        return IsolationForest(**kwargs)

# ===============================
# TensorLy Decompositions (GPU)
# ===============================
from tensorly.decomposition import parafac as _parafac, tucker as _tucker

def decompose_tensor_tucker(tensor, rank):
    """
    GPU-accelerated Tucker via TensorLy (PyTorch backend).
    Input: numpy (H, W, C) -> torch on GPU; Output: numpy core, numpy factors
    """
    T = _to_torch(tensor)
    core_t, factors_t = _tucker(T, rank=rank, init='svd')
    core = _to_numpy(core_t)
    factors = [_to_numpy(f) for f in factors_t]
    return core, factors

def decompose_tensor_parafac(tensor, rank):
    """
    GPU-accelerated CP (PARAFAC) via TensorLy (PyTorch backend).
    Returns: list of factor matrices (numpy)
    """
    T = _to_torch(tensor)
    weights_t, factors_t = _parafac(T, rank=rank, init='svd')
    factors = [_to_numpy(f) for f in factors_t]
    return factors

# ===============================
# Misc
# ===============================
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(1)
warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
# Step 1: Read the data, build the tensor (float32 on read)
# =========================================================
def readData(directory):
    directory = os.fsencode(directory)
    filelist = os.listdir(directory)
    numFiles = len(filelist)
    data_set = np.ones([numFiles, 64, 64, 6], dtype=np.float32)
    true_labels = []
    i = 0
    for file in filelist:
        filename = os.fsdecode(file)

        # If the filename contains _MR_ then it is not an anomaly. If it contains _R#_ it is an anomaly.
        pattern = re.compile(r"_MR_")
        found = bool(pattern.search(filename))
        if found:
            true_labels.append(1)
        else:
            true_labels.append(-1)

        # Build the images into a tensor
        img_array = np.load(os.fsdecode(directory) + "/" + filename).astype(np.float32, copy=False)
        img_array = img_array / 255.0  # Normalize pixel values
        data_set[i, :, :, :] = img_array
        i += 1

    return data_set, np.array(true_labels)

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

# Dataset visualization (optional; comment out if running headless)
# X, true_labels = readData('train_full')
# displayImages(X, {1, 10, 100})
# X, true_labels = readData('test_full')
# displayImages(X, {1, 10, 20})

# ===============================
# Signal plotting (unchanged)
# ===============================
def plot_signals(X, num_sets):
    avgSignalsInSets = []
    for curSet in range(num_sets):
        image_set = X[curSet, :, :, :]
        n_images_in_set = image_set.shape[2]
        signal = np.zeros(image_set.shape[0] * image_set.shape[1], dtype=np.float32)
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

# data_set, true_labels = readData('train_full')
# plot_signals(data_set, 10)

# ===============================
# Feature extractors
# ===============================
def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

# ===============================
# Build tensor (parallel-safe)
# ===============================
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

# ===============================
# Activation flags
# ===============================
use_predefined_rank = False

enable_tucker_oc_svm = False
enable_tucker_autoencoder = True
enable_tucker_random_forest = False
enable_tucker_autoencoder_oc_svm = False

enable_cp_oc_svm = False
enable_cp_autoencoder = False
enable_cp_random_forest = False
enable_cp_autoencoder_oc_svm = False

no_decomposition = False

# ==================================================
# Helpers for manual grid (works for cuML + sklearn)
# ==================================================
def _fit_predict_ocsvm(model, X_train, X_test):
    # cuML expects float32; sklearn works with float64
    if _USE_CUML:
        X_train = X_train.astype(np.float32, copy=False)
        X_test = X_test.astype(np.float32, copy=False)
    model.fit(X_train)
    return model.predict(X_test)

def _best_ocsvm_params(X_train, y_true_like, X_val):
    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    best_acc = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = _make_ocsvm(**params)
        preds = _fit_predict_ocsvm(model, X_train, X_val)
        acc = np.mean(preds == y_true_like)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model
    # re-fit on full train
    best_model.fit(X_train.astype(np.float32 if _USE_CUML else X_train.dtype, copy=False))
    return best_model, best_params, best_acc

def _best_isoforest(X_train, X_val, y_true_like):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.75, 1.0],
        'contamination': [0.05, 0.1, 0.2],
        'max_features': [0.5, 0.75, 1.0]
    }
    best_acc = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = _make_isolation_forest(**params, random_state=42)
        # cuML IF prefers float32
        X_train_cast = X_train.astype(np.float32, copy=False) if _USE_CUML else X_train
        X_val_cast = X_val.astype(np.float32, copy=False) if _USE_CUML else X_val
        model.fit(X_train_cast)
        preds = model.predict(X_val_cast)
        acc = np.mean(preds == y_true_like)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model
    # fit on full
    best_model.fit(X_train.astype(np.float32 if _USE_CUML else X_train.dtype, copy=False))
    return best_model, best_params, best_acc

# ===========================================
# CP decomposition with One-Class SVM (GPU)
# ===========================================
def parafac_OC_SVM(rank, displayConfusionMatrix=False):
    X_train, _ = readData('train_full')
    num_train_sets = X_train.shape[0]

    decomposed_train = buildTensor(X_train, rank, num_train_sets, False)
    features_train = extractFeatures(decomposed_train, num_train_sets, False)
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    X_test, true_labels = readData('test_full')
    num_test_sets = X_test.shape[0]
    decomposed_test = buildTensor(X_test, rank, num_test_sets, False)
    features_test = extractFeatures(decomposed_test, num_test_sets, False)
    features_test_scaled = scaler.transform(features_test)

    # Manual grid to support cuML or sklearn
    model, best_params, best_acc = _best_ocsvm_params(features_train_scaled, true_labels, features_test_scaled)
    print(f"Best accuracy: {best_acc:.3f} with params: {best_params}")

    predictions = model.predict(features_test_scaled.astype(np.float32 if _USE_CUML else features_test_scaled.dtype, copy=False))
    true_labels_arr = np.array(true_labels)
    if set(np.unique(true_labels_arr)) == {0, 1}:
        true_labels_arr = np.where(true_labels_arr == 0, -1, 1)

    accuracy = np.mean(predictions == true_labels_arr)
    print("Accuracy:", accuracy)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels_arr, predictions, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    return accuracy

def ocsvm_raw_geography(displayConfusionMatrix=False):
    X_train, _ = readData('train_full')
    X_test, true_labels = readData('test_full')

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    model, best_params, best_acc = _best_ocsvm_params(X_train_scaled, true_labels, X_test_scaled)
    print(f"Best accuracy: {best_acc:.3f} with params: {best_params}")

    predictions = model.predict(X_test_scaled.astype(np.float32 if _USE_CUML else X_test_scaled.dtype, copy=False))
    accuracy = np.mean(predictions == true_labels)
    print("Accuracy:", accuracy)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, predictions, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    return accuracy, best_params

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

# =====================================
# Tucker with OC-SVM (manual grid)
# =====================================
def tucker_one_class_svm(rank, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets)
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use test set for model selection (your original pipeline did this with GridSearchCV)
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]
    decomposed_test = buildTensor(test_data_set, rank, num_test_sets, True)
    features_test = extractFeatures(decomposed_test, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    model, best_params, _ = _best_ocsvm_params(features_scaled, true_labels, features_scaled_test)
    prediction = model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    accuracy = np.mean(prediction == true_labels)
    print('Best param:', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_one_class_svm():
    print('Tucker rank search One Class SVM')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    for i in rankSet:
        for j in rankSet:
            for k in rankSet:
                rank = (i, j, k)
                accuracy = tucker_one_class_svm(rank)
                rank_accuracy[rank] = accuracy
                print('Rank:', i, j, k, 'Accuracy:', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# CP with autoencoder
# =====================================
def parafac_autoencoder(rank, factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets, False)
    features = extractFeatures(decomposed_data, num_sets, False)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)

    decoder = Dense(128 * factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(features_scaled, features_scaled, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)
    features_test = extractFeatures(decomposed_data, num_test_sets, False)
    features_scaled_test = scaler.transform(features_test)

    reconstructions = autoencoder.predict(features_scaled_test, verbose=0)
    reconstruction_errors = np.mean(np.square(features_scaled_test - reconstructions), axis=1)

    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)

    predictions = (reconstruction_errors > threshold).astype(int)
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1
    accuracy = np.mean(predictions == true_labels)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def autoencoder_anomaly(factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    n_train = X.shape[0]
    X_flat = X.reshape(n_train, -1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_flat)

    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)
    decoder = Dense(128 * factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(features_scaled, features_scaled, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    X_test, true_labels_test = readData('test_full')
    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(n_test, -1)
    features_scaled_test = scaler.transform(X_test_flat)

    reconstructions = autoencoder.predict(features_scaled_test, verbose=0)
    reconstruction_errors = np.mean(np.square(features_scaled_test - reconstructions), axis=1)

    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)

    predictions = (reconstruction_errors > threshold).astype(int)
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1

    accuracy = np.mean(predictions == true_labels_test)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels_test, predictions, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    return accuracy

def autoencoder():
    print('Autoencoder')
    param_accuracy = {}
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            accuracy = autoencoder_anomaly(factor, bottleneck)
            param_accuracy[(factor, bottleneck)] = accuracy
            print('Factor:', factor, 'Bottleneck:', bottleneck, 'Accuracy', accuracy)
    print('Rank accuracy', param_accuracy)
    bestParam = max(param_accuracy, key=param_accuracy.get)
    print('Best param for autoencoder', bestParam, param_accuracy[bestParam])

def cp_rank_search_autoencoder():
    print('CP rank search autoencoder')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            for i in range(startRank, endRank, step):
                rank = i
                accuracy = parafac_autoencoder(rank, factor, bottleneck)
                rank_accuracy[(rank, factor, bottleneck)] = accuracy
                print('Factor:', factor, 'Bottleneck:', bottleneck, 'Rank:', i, 'Accuracy', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# Tucker with autoencoder
# =====================================
def tucker_neural_network_autoencoder(rank, factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets)
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)

    decoder = Dense(128 * factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(features_scaled, features_scaled, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets)
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    reconstructions = autoencoder.predict(features_scaled_test, verbose=0)
    reconstruction_errors = np.mean(np.square(features_scaled_test - reconstructions), axis=1)

    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)

    predictions = (reconstruction_errors > threshold).astype(int)
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1
    accuracy = np.mean(predictions == true_labels)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_autoencoder():
    print('Tucker rank search autoencoder')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    #for factor in range(1, 4):
    for factor in range(2, 4):
        for bottleneck in {16, 32, 64}:
            for i in rankSet:
                for j in rankSet:
                    for k in rankSet:
                        rank = (i, j, k)
                        accuracy = tucker_neural_network_autoencoder(rank, factor, bottleneck)
                        rank_accuracy[(rank, factor, bottleneck)] = accuracy
                        print('Rank:', i, j, k, 'Factor', factor, 'Bottleneck:', bottleneck, 'Accuracy:', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# CP with Isolation Forest
# =====================================
def parafac_random_forest(rank, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets, False)
    features = extractFeatures(decomposed_data, num_sets, False)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)
    features_test = extractFeatures(decomposed_data, num_test_sets, False)
    features_scaled_test = scaler.transform(features_test)

    model, best_params, _ = _best_isoforest(features_scaled, features_scaled_test, true_labels)
    predictions = model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    accuracy = np.mean(predictions == true_labels)
    print('Accuracy:', accuracy)
    print('best parameters', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def isolation_forest_anomaly(displayConfusionMatrix=False):
    X, _ = readData('train_full')
    n_train = X.shape[0]
    X_flat = X.reshape(n_train, -1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_flat)

    X_test, true_labels_test = readData('test_full')
    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(n_test, -1)
    features_scaled_test = scaler.transform(X_test_flat)

    model, best_params, _ = _best_isoforest(features_scaled, features_scaled_test, true_labels_test)
    predictions = model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    accuracy = np.mean(predictions == true_labels_test)
    print('Accuracy:', accuracy)
    print('Best parameters:', best_params)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels_test, predictions, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    return accuracy

def cp_rank_search_random_forest():
    print('CP rank search random forest')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_random_forest(rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# Tucker with Isolation Forest
# =====================================
def tucker_random_forests(rank, displayConfusionMatrix=False):
    print('Running Tucker with Random Forests')
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets)
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, True)
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    model, best_params, _ = _best_isoforest(features_scaled, features_scaled_test, true_labels)
    predictions = model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    accuracy = np.mean(predictions == true_labels)
    print('Accuracy:', accuracy)
    print('best parameters', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_random_forest():
    print('Tucker rank search random forest')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    for i in rankSet:
        for j in rankSet:
            for k in rankSet:
                print('Rank:', i, j, k)
                rank = (i, j, k)
                accuracy = tucker_random_forests(rank)
                rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# CP with autoencoder + OC-SVM
# =====================================
def parafac_autoencoder_oc_svm(rank, factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets, False)
    features = extractFeatures(decomposed_data, num_sets, False)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64 * factor, activation='relu')(encoder)
    bottleneck_layer = Dense(bottleneck, activation='relu')(encoder)

    decoder = Dense(64 * factor, activation='relu')(bottleneck_layer)
    decoder = Dense(128 * factor, activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, shuffle=True, validation_split=0.1,
                    callbacks=[early_stop], verbose=0)

    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
    encoded_features = encoder_model.predict(features_scaled, verbose=0)

    ocsvm_scaler = StandardScaler()
    features_scaled_enc = ocsvm_scaler.fit_transform(encoded_features)

    # Manual param search for OC-SVM on encoded features
    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)
    features_test = extractFeatures(decomposed_data, num_test_sets, False)

    encoded_test_features = encoder_model.predict(features_test, verbose=0)
    features_scaled_test = ocsvm_scaler.transform(encoded_test_features)

    best_acc = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = _make_ocsvm(**params)
        preds = _fit_predict_ocsvm(model, features_scaled_enc, features_scaled_test)
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    print(f"Normal/Anomaly counts are only available after final prediction.")
    prediction = best_model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    normal_count = np.sum(prediction == 1)
    anomaly_count = np.sum(prediction == -1)
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")
    accuracy = np.mean(prediction == true_labels)
    print('Accuracy:', accuracy)
    print('Best params:', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def autoencoder_ocsvm(factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    n_train = X.shape[0]
    X_flat = X.reshape(n_train, -1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_flat)

    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64 * factor, activation='relu')(encoder)
    bottleneck_layer = Dense(bottleneck, activation='relu')(encoder)
    decoder = Dense(64 * factor, activation='relu')(bottleneck_layer)
    decoder = Dense(128 * factor, activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, shuffle=True, validation_split=0.1,
                    callbacks=[early_stop], verbose=0)

    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
    encoded_features = encoder_model.predict(features_scaled, verbose=0)

    ocsvm_scaler = StandardScaler()
    features_scaled_ocsvm = ocsvm_scaler.fit_transform(encoded_features)

    X_test, true_labels = readData('test_full')
    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(n_test, -1)
    features_test_scaled = scaler.transform(X_test_flat)
    encoded_test_features = encoder_model.predict(features_test_scaled, verbose=0)
    features_scaled_test_ocsvm = ocsvm_scaler.transform(encoded_test_features)

    # Manual grid over OC-SVM for compatibility
    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    best_acc = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = _make_ocsvm(**params)
        preds = _fit_predict_ocsvm(model, features_scaled_ocsvm, features_scaled_test_ocsvm)
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    prediction = best_model.predict(features_scaled_test_ocsvm.astype(np.float32 if _USE_CUML else features_scaled_test_ocsvm.dtype, copy=False))
    normal_count = np.sum(prediction == 1)
    anomaly_count = np.sum(prediction == -1)
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")
    accuracy = np.mean(prediction == true_labels)
    print('Accuracy:', accuracy)
    print('Best params:', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    return accuracy

def autoencover_oc_svm():
    print('Autoencoder with oc svm')
    param_accuracy = {}
    for factor in range(1, 4):
        for bottleneck in {16, 32, 64}:
            accuracy = autoencoder_ocsvm(factor, bottleneck)
            param_accuracy[(factor, bottleneck)] = accuracy
            print('Factor:', factor, 'Bottleneck:', bottleneck, 'Accuracy', accuracy)
    print('Rank accuracy', param_accuracy)
    bestParam = max(param_accuracy, key=param_accuracy.get)
    print('Best param for autoencoder with OC-SVM', bestParam, param_accuracy[bestParam])

def cp_rank_search_autoencover_oc_svm():
    print('CP rank search autoencoder with oc svm')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for factor in range(1, 6):
        for bottleneck in sorted({8, 16, 24, 32, 64}):
            for i in range(startRank, endRank, step):
                print('Rank:', i, 'Factor:', factor, 'Bottleneck:', bottleneck)
                rank = i
                accuracy = parafac_autoencoder_oc_svm(rank, factor, bottleneck)
                rank_accuracy[(rank, factor, bottleneck)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# =====================================
# Tucker with autoencoder + OC-SVM
# =====================================
def tucker_autoencoder_ocSVM(rank, factor, bottleneck, displayConfusionMatrix=False):
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    decomposed_data = buildTensor(X, rank, num_sets)

    def flatten_tucker(core, factors):
        return np.concatenate([core.ravel()] + [factor.ravel() for factor in factors], axis=0)

    features = [flatten_tucker(decomposed_data[i][0], decomposed_data[i][1]) for i in range(num_sets)]
    features = np.array(features)

    input_dim = features.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64 * factor, activation='relu')(encoder)
    bottleneck_layer = Dense(bottleneck, activation='relu')(encoder)
    decoder = Dense(64 * factor, activation='relu')(bottleneck_layer)
    decoder = Dense(128 * factor, activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder_output = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
    autoencoder.fit(features, features, epochs=30, batch_size=32, shuffle=True, validation_split=0.1,
                    callbacks=[early_stop], verbose=0)

    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
    encoded_features = encoder_model.predict(features, verbose=0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(encoded_features)

    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, True)
    test_features = [flatten_tucker(decomposed_data[i][0], decomposed_data[i][1]) for i in range(num_test_sets)]
    test_features = np.array(test_features)

    encoded_test_features = encoder_model.predict(test_features, verbose=0)
    features_scaled_test = scaler.transform(encoded_test_features)

    best_acc = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = _make_ocsvm(**params)
        preds = _fit_predict_ocsvm(model, features_scaled, features_scaled_test)
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    prediction = best_model.predict(features_scaled_test.astype(np.float32 if _USE_CUML else features_scaled_test.dtype, copy=False))
    normal_count = np.sum(prediction == 1)
    anomaly_count = np.sum(prediction == -1)
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")
    accuracy = np.mean(prediction == true_labels)
    print('Accuracy:', accuracy)
    print('Best params:', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_autoencoder_OC_svm():
    print('Tucker rank search autoencoder with One Class SVM')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    for factor in range(1, 6):
        for bottleneck in sorted({8, 16, 24, 32, 64}):
            for i in rankSet:
                for j in rankSet:
                    for k in rankSet:
                        rank = (i, j, k)
                        print('Rank:', rank, 'Factor:', factor, 'Bottleneck:', bottleneck)
                        accuracy = tucker_autoencoder_ocSVM(rank, factor, bottleneck)
                        rank_accuracy[(rank, factor, bottleneck)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

# ===============================
# Entrypoints controlled by flags
# ===============================
if __name__ == "__main__":
    if enable_cp_oc_svm:
        if no_decomposition:
            one_class_svm()
        else:
            if not use_predefined_rank:
                bestRank, bestAccuracy = cp_rank_search_one_class_svm()
                print('Best Rank for CP with One Class SVM', bestRank, bestAccuracy)
            else:
                print('Running best rank CP OC-SVM')
                bestRank = 80
                parafac_OC_SVM(bestRank, True)

    if enable_tucker_oc_svm:
        if not use_predefined_rank:
            bestRank, bestAccuracy = tucker_rank_search_one_class_svm()
            print('Tucker Best Rank One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with one-class SVM')
            rank = (5, 5, 35)
            accuracy = tucker_one_class_svm(rank, True)

    if enable_cp_autoencoder:
        if no_decomposition:
            autoencoder()
        else:
            if not use_predefined_rank:
                bestRank, bestAccuracy = cp_rank_search_autoencoder()
                print('Best Rank for CP with autoencoder', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with autoencoder')
                bestRank = 85
                parafac_autoencoder(bestRank, True)

    if enable_tucker_autoencoder:
        if not use_predefined_rank:
            bestRank, bestAccuracy = tucker_rank_search_autoencoder()
            print('Best Rank Tucker with autoencoder', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with autoencoder')
            rank = (5, 5, 35)
            factor = 1
            accuracy = tucker_neural_network_autoencoder(rank, factor, 16, True)

    if enable_cp_random_forest:
        if no_decomposition:
            print('Random forest')
            accuracy = isolation_forest_anomaly()
            print('Random forest accuracy', accuracy)
        else:
            if not use_predefined_rank:
                bestRank, bestAccuracy = cp_rank_search_random_forest()
                print('Best Rank for CP with random forest', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with random forest')
                bestRank = 10
                parafac_random_forest(bestRank, True)

    if enable_tucker_random_forest:
        if not use_predefined_rank:
            bestRank, bestAccuracy = tucker_rank_search_random_forest()
            print('Best Rank Tucker with random forest', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with random forest')
            rank = (5, 65, 5)
            accuracy = tucker_random_forests(rank, True)

    if enable_cp_autoencoder_oc_svm:
        if no_decomposition:
            autoencover_oc_svm()
        else:
            if not use_predefined_rank:
                bestRank, bestAccuracy = cp_rank_search_autoencover_oc_svm()
                print('Best Rank for CP with autoencoder and oc svm', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with autoencoder and oc svm')
                bestRank = 35
                parafac_autoencoder_oc_svm(bestRank, True)

    if enable_tucker_autoencoder_oc_svm:
        if not use_predefined_rank:
            bestRank, bestAccuracy = tucker_rank_search_autoencoder_OC_svm()
            print('Best Rank for Tucker with Autoencoder and One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with autoencoder and one-class SVM')
            rank = (65, 35, 5)
            accuracy = tucker_autoencoder_ocSVM(rank, True)
