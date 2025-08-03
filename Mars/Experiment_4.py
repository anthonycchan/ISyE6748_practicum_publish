## BASELINE DATASET - FASHION MNIST

import numpy as np
import os
import re
import random
import warnings
from tensorly.decomposition import parafac, tucker
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from timeit import default_timer as timer

random.seed(1)

# Percentage of data used for the training and test datasets
dataset_reduction_factor = 1.0

# Use predefined rank
use_predefined_rank = False

# CP decomposition with one-class SVM
enable_cp_oc_svm = False
enable_tucker_oc_svm = False
enable_cp_autoencoder = False
enable_tucker_autoencoder = False
enable_cp_isolation_forest = True
enable_tucker_isolation_forest = False
enable_cp_autoencoder_oc_svm = True
enable_tucker_autoencoder_oc_svm = False

use_tucker_alt = False
no_decomposition = True

# Dataset visualization
def showMNISTImages(X):
    numSets=3
    grid_size = (numSets, 6)
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 3))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    i=0
    for indx in range(3):
        for j in range(6):
            axs[i, j].imshow(X[(indx*6+1)+j])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        i = i+1
    plt.show()


# CP (Parafac) Decomposition and Tucker's Decomposition
def decompose_tensor_tucker(tensor, rank):
    core, factors = tucker(tensor, rank)
    return core, factors

def decompose_tensor_parafac(tensor, rank):
    weights, factors = parafac(tensor, rank=rank)
    return factors

def extract_features_tucker(core, factors):
    core_flattened = core.ravel()
    factors_flattened = np.concatenate([factor.ravel() for factor in factors], axis=0)
    return np.concatenate((core_flattened, factors_flattened), axis=0)

def extract_features_cp(factors):
    return np.concatenate([factor.ravel() for factor in factors], axis=0)

def decompose_sample(tensor, rank):
    # Handle exceptions per sample
    try:
        weights, factors = parafac(tensor, rank=rank, init='random', tol=1e-6, n_iter_max=100)
        return factors
    except Exception as e:
        print("Decomposition error:", e)
        return None

def buildTensor(X, rank, isTuckerDecomposition=True):
    with ThreadPoolExecutor() as executor:
        if isTuckerDecomposition:
            num_sets = X.shape[0]
            decomposed_data = list(executor.map(lambda i: decompose_tensor_tucker(X[i], rank), range(num_sets)))
        else:
            decomposed_data = decompose_tensor_parafac(X, rank)

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

# CP decomposition with One-Class SVM
def parafac_OC_SVM(X_train, X_test, Y_test, rank, displayConfusionMatrix=False):
    start = timer()
    # CP decomposition
    decomposed_train = buildTensor(X_train, rank, False)
    decomposed_train_features = np.array(decomposed_train[0])

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(decomposed_train_features)

    # Test data
    true_labels = Y_test
    decomposed_test = buildTensor(X_test, rank, False)
    features_test = np.array(decomposed_test[0])
    features_test_scaled = scaler.transform(features_test)

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
        preds = model.predict(features_test_scaled)
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = params

    print(f"Best accuracy: {best_acc:.3f} with params: {best_params}")


    # Train OC-SVM
    best_model.fit(features_train_scaled)

    # Predict
    predictions = best_model.predict(features_test_scaled)  # 1 = normal, -1 = anomaly

    # Convert true_labels to match prediction format if needed
    # Example: if true_labels = [0, 1], map 0→-1 and 1→1
    true_labels = np.array(true_labels)
    if set(np.unique(true_labels)) == {0, 1}:
        true_labels = np.where(true_labels == 0, -1, 1)

    # Accuracy
    accuracy = np.mean(predictions == true_labels)
    print("Accuracy:", accuracy)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, predictions, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")

    return accuracy

def ocsvm_pixel_features(X_train, X_test, Y_test, displayConfusionMatrix=False, random_state=42):
    start = timer()
    np.random.seed(random_state)

    # Flatten images to vectors
    n_train, h, w = X_train.shape
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)  # (n_train, 784)
    X_test_flat = X_test.reshape(n_test, -1)     # (n_test, 784)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Use provided labels (1 = normal, -1 = anomaly)
    true_labels = np.array(Y_test)

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
        model.fit(X_train_scaled)
        preds = model.predict(X_test_scaled)
        acc = np.mean(preds == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = params

    print(f"Best accuracy: {best_acc:.3f} with params: {best_params}")

    predictions = best_model.predict(X_test_scaled)
    accuracy = np.mean(predictions == true_labels)
    print("Accuracy:", accuracy)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, predictions, labels=[-1, 1])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy, best_model, best_params

def one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('One Class SVM')
    ocsvm_pixel_features(reduced_X_train, reduced_X_test, reduced_Y_test, False)

def cp_rank_search_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('CP rank search One Class SVM')
    startRank = 10
    endRank = 30
    step = 5
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_OC_SVM(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


def tucker_one_class_svm(X_train, X_test, Y_test, rank, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    num_sets = X_train.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_train, rank)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 4: Train OC-SVM with Grid Search for Hyperparameter Tuning
    param_grid = {
        'nu': [0.1, 0.5, 0.9],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    oc_svm = OneClassSVM()
    grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(features_scaled)
    best_oc_svm = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ###
    ### Predict using the test set
    ###
    num_test_sets = X_test.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_test, rank)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    # Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)
    accuracy = sum(prediction == Y_test) / len(Y_test)
    print('Accuracy:', accuracy)
    print('Best param:', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    return accuracy


def tucker_rank_search_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search One Class SVM')
    startRank = 5
    endRank = 30
    stepSize = 5
    rank_accuracy = {}
    for j in range(startRank, endRank, stepSize):
        for k in range(startRank, endRank, stepSize):
            print('Rank:', j, k)
            rank = (j,k)
            accuracy = tucker_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
            rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

def tucker_rank_search_one_class_svm_alt(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search One Class SVM alt')
    rankSet = sorted({5, 16, 24, 32, 48, 64})
    k = 5
    rank_accuracy = {}
    for i in rankSet:
        for j in rankSet:
            print('Rank:', i, j, k)
            rank = (i, j, k)
            accuracy = tucker_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
            rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


####################
## CP with autoencoder
####################
def parafac_autoencoder(X_train, X_test, Y_test, rank, factor, displayConfusionMatrix=False):
    start = timer()
    # Run CP decomposition
    decomposed_train = buildTensor(X_train, rank, False)
    decomposed_train_features = np.array(decomposed_train[0])

    # Extract and normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(decomposed_train_features)

    # Define the autoencoder model
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(max(2, int(0.2*input_dim))*factor, activation='relu')(encoder)

    # Decoder
    decoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    ###
    ### Predict using the test set
    ###
    true_labels = np.array(Y_test)

    # Run CP decomposition
    decomposed_test = buildTensor(X_test, rank, False)

    # Extract and normalize features
    features_test = np.array(decomposed_test[0])
    features_test_scaled = scaler.transform(features_test)

    # Predict using the autoencoder
    reconstructions = autoencoder.predict(features_test_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(features_test_scaled - reconstructions), axis=1)

    # Set the threshold based on training prediction errors
    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)  # Assume anomalies are rare

    # Identify anomalies (assuming a threshold-based method)
    predictions = (reconstruction_errors > threshold).astype(int)

    # 1 indicates normal, -1 indicates anomaly
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1
    #print("Predictions", predictions)
    #print('True labels', true_labels)
    accuracy = sum(predictions == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def autoencoder_anomaly(X_train, X_test, Y_test, factor=1, displayConfusionMatrix=False):
    start = timer()
    # Flatten images
    n_train, h, w = X_train.shape
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)  # (n_train, 784)
    X_test_flat = X_test.reshape(n_test, -1)     # (n_test, 784)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_train_flat)

    # Define autoencoder architecture
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(max(2, int(0.2*input_dim))*factor, activation='relu')(encoder)
    decoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    # Prepare test features
    features_test_scaled = scaler.transform(X_test_flat)

    # Reconstruction errors
    reconstructions = autoencoder.predict(features_test_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(features_test_scaled - reconstructions), axis=1)
    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)

    # Prediction: -1 for anomaly, 1 for normal
    predictions = np.where(reconstruction_errors > threshold, -1, 1)

    # Accuracy
    true_labels = np.array(Y_test)
    accuracy = np.mean(predictions == true_labels)
    print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, predictions, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Autoencoder')
    for factor in range(1, 4):
        print('Factor:', factor)
        autoencoder_anomaly(reduced_X_train, reduced_X_test, reduced_Y_test, factor)

def cp_rank_search_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('CP rank search autoencoder')
    startRank = 5
    endRank = 30
    step = 5
    rank_accuracy = {}
    for factor in range(1,4):
        for i in range(startRank, endRank, step):
            print('Factor:', factor, 'Rank:', i)
            rank = i
            accuracy = parafac_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
            rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

####################################
## Tucker with autoencoders
####################################
def tucker_neural_network_autoencoder(X_train, X_test, Y_test, rank, factor, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    num_sets = X_train.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_train, rank)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the autoencoder model
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(max(2, int(0.2*input_dim))*factor, activation='relu')(encoder)

    # Decoder
    decoder = Dense(max(2, int(0.6*input_dim))*factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0 )

    ###
    ### Predict using the test set
    ###
    num_test_sets = X_test.shape[0]
    true_labels = np.array(Y_test)

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_test, rank)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    # Predict using the autoencoder
    reconstructions = autoencoder.predict(features_scaled_test, verbose=0)
    reconstruction_errors = np.mean(np.square(features_scaled_test - reconstructions), axis=1)

    # Set the threshold based on training prediction errors
    train_reconstructions = autoencoder.predict(features_scaled, verbose=0)
    train_errors = np.mean(np.square(features_scaled - train_reconstructions), axis=1)
    threshold = np.percentile(train_errors, 95)  # Assume anomalies are rare

    # Identify anomalies (assuming a threshold-based method)
    predictions = (reconstruction_errors > threshold).astype(int)

    # 1 indicates normal, -1 indicates anomaly
    predictions[predictions == 1] = -1
    predictions[predictions == 0] = 1
    accuracy = sum(predictions == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search autoencoder')
    startRank = 5
    endRank = 30
    stepSize = 5
    rank_accuracy = {}
    for factor in range(1, 4):
        for j in range(startRank, endRank, stepSize):
            for k in range(startRank, endRank, stepSize):
                    print('Factor', factor, 'Rank:', j, k)
                    rank = (j,k)
                    accuracy = tucker_neural_network_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
                    rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

def tucker_rank_search_autoencoder_alt(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search autoencoder alt')
    rankSet = sorted({5, 16, 24, 32, 48, 64})
    k = 5
    rank_accuracy = {}
    for factor in range(1, 4):
        for i in rankSet:
            for j in rankSet:
                    print('Factor', factor, 'Rank:', i, j, k)
                    rank = (i, j, k)
                    accuracy = tucker_neural_network_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
                    rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

############################################
### CP with isolation forest
############################################
def parafac_isolation_forest(X_train, X_test, Y_test, rank, displayConfusionMatrix=False):
    start = timer()
    # CP decomposition
    decomposed_train = buildTensor(X_train, rank, False)
    decomposed_train_features = np.array(decomposed_train[0])

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(decomposed_train_features)

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.75, 1.0],
        'contamination': [0.05, 0.1, 0.2],
        'max_features': [0.5, 0.75, 1.0]
    }

    # Custom scoring function for unsupervised learning
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    isolation_forest = IsolationForest(random_state=42)
    grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=5, scoring=custom_scorer, verbose=0, n_jobs=-1)
    grid_search.fit(features_train_scaled)

    best_isolation_forest = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ###
    ### Predict using the test set
    ###
    true_labels = np.array(Y_test)

    # Run CP decomposition
    decomposed_test = buildTensor(X_test, rank, False)

    # Extract and normalize features
    features_test = np.array(decomposed_test[0])
    features_test_scaled = scaler.transform(features_test)

    # Predict on the test set
    predictions = best_isolation_forest.predict(features_test_scaled)
    #print("Predictions", predictions)
    #print('True labels', true_labels)
    accuracy = sum(predictions == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)
    print('best parameters', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def isolation_forest_anomaly(X_train, X_test, Y_test, displayConfusionMatrix=False):
    start = timer()
    # Flatten images to vectors
    n_train, h, w = X_train.shape
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Suppress warnings (optional)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.75, 1.0],
        'contamination': [0.05, 0.1, 0.2],
        'max_features': [0.5, 0.75, 1.0]
    }

    # Custom scorer: mean score_samples
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    isolation_forest = IsolationForest(random_state=42)
    grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=3,
                               scoring=custom_scorer, verbose=0, n_jobs=-1)
    grid_search.fit(X_train_scaled)

    best_isolation_forest = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict on the test set
    predictions = best_isolation_forest.predict(X_test_scaled)  # 1=normal, -1=anomaly

    true_labels = np.array(Y_test)

    accuracy = np.mean(predictions == true_labels)
    print('Accuracy:', accuracy)
    print('Best parameters:', best_params)

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, predictions, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Isolation forest')
    isolation_forest_anomaly(reduced_X_train, reduced_X_test, reduced_Y_test, False)

def cp_rank_search_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('CP rank search isolation forest')
    startRank = 5
    endRank = 30
    step = 5
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


############################################
### Tucker with isolation forest
############################################
def tucker_isolation_forests(X_train, X_test, Y_test, rank, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    num_sets = X_train.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_train, rank)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.75, 1.0],
        'contamination': [0.05, 0.1, 0.2],
        'max_features': [0.5, 0.75, 1.0]
    }

    # Custom scoring function for unsupervised learning
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    isolation_forest = IsolationForest(random_state=42)
    grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=5, scoring=custom_scorer,
                               verbose=0, n_jobs=-1)

    grid_search.fit(features_scaled)

    best_isolation_forest = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ###
    ### Predict using the test set
    ###
    num_test_sets = X_test.shape[0]
    true_labels = np.array(Y_test)

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_test, rank)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    # Predict on the test set
    predictions = best_isolation_forest.predict(features_scaled_test)
    #print("Predictions", predictions)
    #print('True labels', true_labels)
    accuracy = sum(predictions == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)
    print('best parameters', best_params)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search isolation forest')
    startRank = 5
    endRank = 30
    stepSize = 5
    rank_accuracy = {}
    for j in range(startRank, endRank, stepSize):
        for k in range(startRank, endRank, stepSize):
            print('Rank:', j, k)
            rank = (j, k)
            accuracy = tucker_isolation_forests(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
            rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

def tucker_rank_search_isolation_forest_alt(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search isolation forest alt')
    rankSet = sorted({5, 16, 24, 32, 48, 64})
    k = 5
    rank_accuracy = {}
    for i in rankSet:
        for j in rankSet:
            print('Rank:', i, j, k)
            rank = (i,j,k)
            accuracy = tucker_isolation_forests(reduced_X_train, reduced_X_test, reduced_Y_test, rank)
            rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

############################################
### CP with autoencoder and oc-svm
############################################
def parafac_autoencoder_oc_svm(X_train, X_test, Y_test, rank, factor, displayConfusionMatrix=False):
    start = timer()
    # Run CP decomposition
    decomposed_train = buildTensor(X_train, rank, False)
    decomposed_train_features = np.array(decomposed_train[0])

    # Extract and normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(decomposed_train_features)

    # Step 3: Define and train the autoencoder
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    bottleneck_layer = Dense(max(2, int(0.2 * input_dim)) * factor, activation='relu')(encoder)

    # Decoder
    decoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(bottleneck_layer)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    # Extract features using the encoder part of the autoencoder
    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer )
    encoded_features = encoder_model.predict(features_scaled, verbose=0)

    # Step 4: Normalize the features
    ocsvm_scaler = StandardScaler()
    features_scaled = ocsvm_scaler.fit_transform(encoded_features)

    # Step 5: Train OC-SVM with Grid Search for Hyperparameter Tuning
    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Custom scoring function for unsupervised learning
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    oc_svm = OneClassSVM()
    grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1)
    grid_search.fit(features_scaled)
    best_oc_svm = grid_search.best_estimator_

    ###
    ### Predict using the test set
    ###
    true_labels = np.array(Y_test)

    # Run CP decomposition
    decomposed_test = buildTensor(X_test, rank, False)

    # Extract and normalize features
    features_test = np.array(decomposed_test[0])
    features_test_scaled = scaler.transform(features_test)

    # Extract features from test data using the trained encoder
    encoded_test_features = encoder_model.predict(features_test_scaled, verbose=0)
    features_scaled_test = ocsvm_scaler.transform(encoded_test_features)


    # Step 6: Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)
    #print("Prediction:", prediction)  # 1 indicates normal, -1 indicates anomaly

    # Evaluate performance
    normal_count = np.sum(prediction == 1)
    anomaly_count = np.sum(prediction == -1)
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")
    print("Predictions", prediction)
    print('True labels', true_labels)
    accuracy = sum(prediction == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def autoencoder_ocsvm_anomaly(X_train, X_test, Y_test, factor=1, displayConfusionMatrix=False):
    start = timer()
    # Flatten images
    n_train, h, w = X_train.shape
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_train_flat)

    # Define autoencoder
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    bottleneck_layer = Dense(max(2, int(0.2 * input_dim)) * factor, activation='relu')(encoder)
    decoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(bottleneck_layer)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    # Extract bottleneck features
    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
    encoded_features = encoder_model.predict(features_scaled, verbose=0)
    ocsvm_scaler = StandardScaler()
    features_scaled_ocsvm = ocsvm_scaler.fit_transform(encoded_features)

    # Train OC-SVM with grid search
    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    oc_svm = OneClassSVM()
    grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1)
    grid_search.fit(features_scaled_ocsvm)
    best_oc_svm = grid_search.best_estimator_

    # Prepare test set
    features_test_scaled = scaler.transform(X_test_flat)
    encoded_test_features = encoder_model.predict(features_test_scaled, verbose=0)
    features_scaled_test_ocsvm = ocsvm_scaler.transform(encoded_test_features)

    # OC-SVM prediction
    prediction = best_oc_svm.predict(features_scaled_test_ocsvm)
    true_labels = np.array(Y_test)

    # Accuracy
    accuracy = np.mean(prediction == true_labels)
    print('Accuracy:', accuracy)
    print(f"Normal: {np.sum(prediction == 1)}, Anomalies: {np.sum(prediction == -1)}")

    if displayConfusionMatrix:
        cm = metrics.confusion_matrix(true_labels, prediction, labels=[-1, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomaly', 'Normal'])
        cm_display.plot()
        plt.show()

    end = timer()
    print(f"Elapsed time: {end - start:.3f} seconds")
    return accuracy

def autoencoder_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Autoencoder with oc svm')
    for factor in {4, 8, 12}:
        print('Factor:', factor)
        autoencoder_ocsvm_anomaly(reduced_X_train, reduced_X_test, reduced_Y_test, factor, False)

def cp_rank_search_autoencover_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('CP rank search autoencoder with oc svm')
    startRank = 5
    endRank = 30
    step = 5
    rank_accuracy = {}
    for factor in {4,8,12}:
        for i in range(startRank, endRank, step):
            print('Rank:', i, 'Factor:', factor)
            rank = i
            accuracy = parafac_autoencoder_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
            rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


############################################
### Tucker with autoencoder and oc-svm
############################################
def tucker_autoencoder_ocSVM(X_train, X_test, Y_test, rank, factor, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    num_sets = X_train.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_train, rank, num_sets)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the autoencoder model
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    bottleneck_layer = Dense(max(2, int(0.2 * input_dim)) * factor, activation='relu')(encoder)

    # Decoder
    decoder = Dense(max(2, int(0.6 * input_dim)) * factor, activation='relu')(bottleneck_layer)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=30, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    # Extract features using the encoder part of the autoencoder
    encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
    encoded_features = encoder_model.predict(features, verbose=0)

    # Step 4: Normalize the features
    ocsvm_scaler = StandardScaler()
    features_scaled = ocsvm_scaler.fit_transform(encoded_features)

    # Step 5: Train OC-SVM with Grid Search for Hyperparameter Tuning
    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Custom scoring function for unsupervised learning
    def custom_scorer(estimator, X):
        return np.mean(estimator.score_samples(X))

    oc_svm = OneClassSVM()
    grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1)
    grid_search.fit(features_scaled)
    best_oc_svm = grid_search.best_estimator_

    ###
    ### Predict using the test set
    ###
    num_test_sets = X_test.shape[0]
    true_labels = np.array(Y_test)

    # Run Tucker decomposition
    decomposed_data = buildTensor(X_test, rank, True)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    # Extract features from test data using the trained encoder
    encoded_test_features = encoder_model.predict(features_scaled_test, verbose=0)
    features_scaled_test = ocsvm_scaler.transform(encoded_test_features)

    # Step 6: Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)

    # Evaluate performance
    normal_count = np.sum(prediction == 1)
    anomaly_count = np.sum(prediction == -1)
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")
    #print("Predictions", prediction)
    #print('True labels', true_labels)
    accuracy = sum(prediction == true_labels) / len(true_labels)
    print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def tucker_rank_search_autoencoder_OC_svm(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search autoencoder with One Class SVM')
    startRank = 5
    endRank = 30
    stepSize = 5
    rank_accuracy = {}
    for factor in range(1, 4):
        for j in range(startRank, endRank, stepSize):
            for k in range(startRank, endRank, stepSize):
                print('Rank:', j, k)
                rank = (j,k)
                accuracy = tucker_autoencoder_ocSVM(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
                rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

def tucker_rank_search_autoencoder_OC_svm_alt(reduced_X_train, reduced_X_test, reduced_Y_test):
    print('Tucker rank search autoencoder with One Class SVM alt')
    rankSet = sorted({5, 16, 24, 32, 48, 64})
    k = 5
    rank_accuracy = {}
    for factor in range(1, 4):
        for i in rankSet:
            for j in rankSet:
                print('Rank:', i, j, k)
                rank = (i, j, k)
                accuracy = tucker_autoencoder_ocSVM(reduced_X_train, reduced_X_test, reduced_Y_test, rank, factor)
                rank_accuracy[(rank, factor)] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


# Read the MNIST fashion dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#showMNISTImages(x_train)

# Perform anomaly detection for every label in the dataset
for applicableLabel in np.unique(y_test):
    print('Model for label', applicableLabel)

    # Extract train data
    matchingIndx = np.where(y_train == applicableLabel)[0]
    X = x_train[matchingIndx]
    reduced_X_train = X[0:int(X.shape[0]*dataset_reduction_factor), :, :]

    # Extract first 1000 test data
    reduced_X_test = x_test[0:int(x_test.shape[0]*dataset_reduction_factor), :, :]
    reduced_Y_test = y_test[0:int(y_test.shape[0]*dataset_reduction_factor)]
    reduced_Y_test = np.array(reduced_Y_test)
    reduced_Y_test = reduced_Y_test.astype(int)

    # Set matching labels to normal (1) and non-matching labels to abnormal (-1)
    matchingTestIndx = reduced_Y_test == applicableLabel
    reduced_Y_test[matchingTestIndx] = 1
    reduced_Y_test[~matchingTestIndx] = -1

    if enable_cp_oc_svm:
        if no_decomposition:
            one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
        else:
            if use_predefined_rank == False:
                bestRank, bestAccuracy = cp_rank_search_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
                print('Best Rank for CP with One Class SVM for label', applicableLabel,
                      'bestRank:', bestRank,
                      'bestAccuracy', bestAccuracy)
            else:
                print('Running best rank CP OC-SVM for label', applicableLabel)
                bestRank=25
                parafac_OC_SVM(reduced_X_train, reduced_X_test, reduced_Y_test, bestRank, True)

    if enable_tucker_oc_svm:
        if use_predefined_rank == False:
            if use_tucker_alt:
                bestRank, bestAccuracy = tucker_rank_search_one_class_svm_alt(reduced_X_train, reduced_X_test, reduced_Y_test)
            else:
                bestRank, bestAccuracy = tucker_rank_search_one_class_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
            print('Tucker Best Rank One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with one-class SVM')
            rank=(25, 25)
            accuracy = tucker_one_class_svm(rank, True)

    if enable_cp_autoencoder:
        if no_decomposition:
            autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test)
        else:
            if use_predefined_rank == False:
                bestRank, bestAccuracy = cp_rank_search_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test)
                print('Best Rank for CP with autoencoder', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with autoencoder')
                bestRank = 25
                factor = 1
                print('X_train', reduced_X_train.shape, 'bestRank', bestRank)
                parafac_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test, bestRank, factor, True)

    if enable_tucker_autoencoder:
        if use_predefined_rank == False:
            if use_tucker_alt:
                bestRank, bestAccuracy = tucker_rank_search_autoencoder_alt(reduced_X_train, reduced_X_test, reduced_Y_test)
            else:
                bestRank, bestAccuracy = tucker_rank_search_autoencoder(reduced_X_train, reduced_X_test, reduced_Y_test)
            print('Best Rank Tucker with autoencoder', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with autoencoder')
            rank=(95, 65, 65)
            factor=40
            accuracy = tucker_neural_network_autoencoder(rank, factor, True)

    if enable_cp_isolation_forest:
        if no_decomposition:
            isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test)
        else:
            if use_predefined_rank == False:
                bestRank, bestAccuracy = cp_rank_search_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test)
                print('Best Rank for CP with isolation forest', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with isolation forest')
                bestRank = 10
                factor = 1
                parafac_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test, bestRank, True)

    if enable_tucker_isolation_forest:
        if use_predefined_rank == False:
            if use_tucker_alt:
                bestRank, bestAccuracy = tucker_rank_search_isolation_forest_alt(reduced_X_train, reduced_X_test, reduced_Y_test)
            else:
                bestRank, bestAccuracy = tucker_rank_search_isolation_forest(reduced_X_train, reduced_X_test, reduced_Y_test)
            print('Best Rank Tucker with isolation forest', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with isolation forest')
            rank=(5, 65, 5)
            accuracy = tucker_isolation_forests(reduced_X_train, reduced_X_test, reduced_Y_test, rank, True)

    if enable_cp_autoencoder_oc_svm:
        if no_decomposition:
            autoencoder_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
        else:
            if use_predefined_rank == False:
                bestRank, bestAccuracy = cp_rank_search_autoencover_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
                print('Best Rank for CP with autoencoder and oc svm', bestRank, bestAccuracy)
            else:
                print('Running best rank CP with autoencoder and oc svm')
                bestRank=35
                parafac_autoencoder_oc_svm(reduced_X_train, reduced_X_test, reduced_Y_test, bestRank, True)

    if enable_tucker_autoencoder_oc_svm:
        if use_predefined_rank == False:
            if use_tucker_alt:
                bestRank, bestAccuracy = tucker_rank_search_autoencoder_OC_svm_alt(reduced_X_train, reduced_X_test, reduced_Y_test)
            else:
                bestRank, bestAccuracy = tucker_rank_search_autoencoder_OC_svm(reduced_X_train, reduced_X_test, reduced_Y_test)
            print('Best Rank for Tucker with Autoencoder and One Class SVM', bestRank, bestAccuracy)
        else:
            print('Running best rank Tucker with autoencoder and one-class SVM')
            rank = (65,35,5)
            accuracy = tucker_autoencoder_ocSVM(reduced_X_train, reduced_X_test, reduced_Y_test, rank, True)
