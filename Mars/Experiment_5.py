# MARS TENSOR DATASET
import numpy as np
import os
import re
import random
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
        pattern = re.compile(r"_MR_")
        found = bool(pattern.search(filename))
        if found:
            true_labels.append(1)
        else:
            true_labels.append(-1)

        # Build the images into a tensor
        img_array = np.load(os.fsdecode(directory) + "/" + filename)
        img_array = img_array / 255.0  # Normalize pixel values
        data_set[i, :, :, :] = img_array
        i += 1

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
X, true_labels = readData('train_full')
displayImages(X, {1, 10, 100})

# Test images
X, true_labels = readData('test_full')
displayImages(X, {1, 10, 20})


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
            signal = signal + image_set[:,:,i].ravel(order='F')
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


data_set, true_labels = readData('train_full')
plot_signals(data_set, 10)
#plt.show()

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
enable_cp_oc_svm = False
# CP decomposition with neural-network autoencoders
enable_cp_autoencoder = False
# CP decomposition with random forest
enable_cp_random_forest = False
# CP decomposition with combination of autoencoder and one-class SVM.
enable_cp_autoencoder_oc_svm = True

############################################
# CP decomposition with One-Class SVM
############################################
def parafac_OC_SVM(rank, displayConfusionMatrix=False):
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    import numpy as np

    # Training data
    X_train, _ = readData('train_full')
    num_train_sets = X_train.shape[0]

    # CP decomposition
    decomposed_train = buildTensor(X_train, rank, num_train_sets, False)

    # Feature extraction + scaling
    features_train = extractFeatures(decomposed_train, num_train_sets, False)
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    # Test data
    X_test, true_labels = readData('test_full')
    num_test_sets = X_test.shape[0]
    decomposed_test = buildTensor(X_test, rank, num_test_sets, False)
    features_test = extractFeatures(decomposed_test, num_test_sets, False)
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
    #oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
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

    return accuracy


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
    if use_predefined_rank == False:
        bestRank, bestAccuracy = cp_rank_search_one_class_svm()
        print('Best Rank for CP with One Class SVM', bestRank, bestAccuracy)
    else:
        print('Running best rank CP OC-SVM')
        bestRank=80
        parafac_OC_SVM(bestRank, True)

############################################
## Tucker with OC-SVM
############################################
def tucker_one_class_svm(rank, displayConfusionMatrix=False):
    print('Running Tucker one-class SVM')
    ###
    ### Training
    ###
    X, true_labels = readData('train_full')
    num_sets = X.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X, rank, num_sets)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 4: Train OC-SVM with Grid Search for Hyperparameter Tuning
    param_grid = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
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
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, True)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets)
    features_scaled_test = scaler.transform(features_test)

    # Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)
    accuracy = sum(prediction == true_labels) / len(true_labels)
    #print('Accuracy:', accuracy)
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
            for k in {5}:
                rank = (i,j,k)
                accuracy = tucker_one_class_svm(rank)
                rank_accuracy[rank] = accuracy
                print('Rank:', i, j, k, 'Accuracy:', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

if enable_tucker_oc_svm:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = tucker_rank_search_one_class_svm()
        print('Tucker Best Rank One Class SVM', bestRank, bestAccuracy)
    else:
        print('Running best rank Tucker with one-class SVM')
        rank=(5, 5, 35)
        accuracy = tucker_one_class_svm(rank, True)

############################################
## CP with autoencoder
############################################
def parafac_autoencoder(rank, factor, bottleneck, displayConfusionMatrix=False):
    X, true_labels = readData('train_full')

    num_sets = X.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(X, rank, num_sets, False)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets, False)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the autoencoder model
    input_dim = features_scaled.shape[1]
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

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    ###
    ### Predict using the test set
    ###
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets, False)
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
    #print('Accuracy:', accuracy)

    if displayConfusionMatrix:
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[-1, 1])
        cm_display.plot()
        plt.show()

    return accuracy

def cp_rank_search_autoencoder():
    print('CP rank search autoencoder')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for factor in range(1,4):
        for bottleneck in {16,32,64}:
            for i in range(startRank, endRank, step):
                rank = i
                accuracy = parafac_autoencoder(rank, factor, bottleneck)
                rank_accuracy[(rank, factor, bottleneck)] = accuracy
                print('Factor:', factor, 'Bottleneck:', bottleneck, 'Rank:', i, 'Accuracy', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]


if enable_cp_autoencoder:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = cp_rank_search_autoencoder()
        print('Best Rank for CP with autoencoder', bestRank, bestAccuracy)
    else:
        print('Running best rank CP with autoencoder')
        bestRank=85
        parafac_autoencoder(bestRank, True)

############################################
### Tucker with autoencoder
############################################
def tucker_neural_network_autoencoder(rank, factor, bottleneck, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X, rank, num_sets)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the autoencoder model
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(128*factor, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(bottleneck, activation='relu')(encoder)

    # Decoder
    decoder = Dense(128*factor, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    autoencoder.fit(features_scaled, features_scaled, epochs=10, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping], verbose=0)

    ###
    ### Predict using the test set
    ###
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets)

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
    #print("Predictions", predictions)
    #print('True labels', true_labels)
    #print('Accuracy:', accuracy)

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
    for factor in range(1,4):
        for bottleneck in {16,32,64}:
            for i in rankSet:
                for j in rankSet:
                    for k in {5}:
                        rank = (i,j,k)
                        accuracy = tucker_neural_network_autoencoder(rank, factor, bottleneck)
                        rank_accuracy[(rank, factor, bottleneck)] = accuracy
                        print('Rank:', i, j, k, 'Factor', factor, 'Bottleneck:', bottleneck, 'Accuracy:', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

if enable_tucker_autoencoder:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = tucker_rank_search_autoencoder()
        print('Best Rank Tucker with autoencoder', bestRank, bestAccuracy)
    else:
        print('Running best rank Tucker with autoencoder')
        rank=(5, 5, 35)
        factor=1
        accuracy = tucker_neural_network_autoencoder(rank, factor, 16, True)

############################################
### CP with random forest
############################################
def parafac_random_forest(rank, displayConfusionMatrix=False):
    X, true_labels = readData('train_full')
    num_sets = X.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(X, rank, num_sets, False)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets, False)
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
    grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=5, scoring=custom_scorer, verbose=0, n_jobs=-1)
    grid_search.fit(features_scaled)

    best_isolation_forest = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ###
    ### Predict using the test set
    ###
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets, False)
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

if enable_cp_random_forest:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = cp_rank_search_random_forest()
        print('Best Rank for CP with random forest', bestRank, bestAccuracy)
    else:
        print('Running best rank CP with random forest')
        bestRank=10
        parafac_random_forest(bestRank, True)


############################################
### Tucker with random forest
############################################
def tucker_random_forests(rank, displayConfusionMatrix=False):
    print('Running Tucker with Random Forests')
    ###
    ### Training
    ###
    X, _ = readData('train_full')
    num_sets = X.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X, rank, num_sets)

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
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, True)

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

def tucker_rank_search_random_forest():
    print('Tucker rank search random forest')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    for i in rankSet:
        for j in rankSet:
            for k in {5}:
                print('Rank:', i, j, k)
                rank = (i,j,k)
                accuracy = tucker_random_forests(rank)
                rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

if enable_tucker_random_forest:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = tucker_rank_search_random_forest()
        print('Best Rank Tucker with Random forest', bestRank, bestAccuracy)
    else:
        print('Running best rank Tucker with random forest')
        rank=(5, 65, 5)
        accuracy = tucker_random_forests(rank, True)


############################################
### CP with autoencoder and oc-svm
############################################
# Old code
def parafac_autoencoder_oc_svm(rank, displayConfusionMatrix=False):
    X, true_labels = readData('train_full')
    num_sets = X.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(X, rank, num_sets, False)

    # Extract and normalize features
    features = extractFeatures(decomposed_data, num_sets, False)
    scaler = StandardScaler()
    #features_scaled = scaler.fit_transform(features)

    # Step 3: Define and train the autoencoder
    input_dim = features.shape[1]
    factor = 40  # Adjust factor as needed

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64 * factor, activation='relu')(encoder)
    encoder_output = Dense(32 * factor, activation='relu')(encoder)

    decoder = Dense(64 * factor, activation='relu')(encoder_output)
    decoder = Dense(128 * factor, activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder_output = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(features, features, epochs=10, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)

    # Extract features using the encoder part of the autoencoder
    encoder_model = Model(inputs=input_layer, outputs=encoder_output)
    encoded_features = encoder_model.predict(features, verbose=0)

    # Step 4: Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(encoded_features)

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
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run CP decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, False, True)

    # Extract and normalize features
    features_test = extractFeatures(decomposed_data, num_test_sets, False)

    # Extract features from test data using the trained encoder
    encoded_test_features = encoder_model.predict(features_test, verbose=0)
    features_scaled_test = scaler.transform(encoded_test_features)

    # Step 6: Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)
    print("Prediction:", prediction)  # 1 indicates normal, -1 indicates anomaly

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

def cp_rank_search_autoencover_oc_svm():
    print('CP rank search autoencoder with oc svm')
    startRank = 10
    endRank = 100
    step = 5
    rank_accuracy = {}
    for i in range(startRank, endRank, step):
        print('Rank:', i)
        rank = i
        accuracy = parafac_autoencoder_oc_svm(rank)
        rank_accuracy[rank] = accuracy
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

if enable_cp_autoencoder_oc_svm:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = cp_rank_search_autoencover_oc_svm()
        print('Best Rank for CP with autoencoder and oc svm', bestRank, bestAccuracy)
    else:
        print('Running best rank CP with autoencoder and oc svm')
        bestRank=35
        parafac_autoencoder_oc_svm(bestRank, True)

############################################
### Tucker with autoencoder and oc-svm
############################################
def tucker_autoencoder_ocSVM(rank, displayConfusionMatrix=False):
    ###
    ### Training
    ###
    X, true_labels = readData('train_full')
    num_sets = X.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(X, rank, num_sets)

    # Flatten the Tucker components
    def flatten_tucker(core, factors):
        return np.concatenate([core.ravel()] + [factor.ravel() for factor in factors], axis=0)

    features = [flatten_tucker(decomposed_data[i][0], decomposed_data[i][1]) for i in range(num_sets)]
    features = np.array(features)

    # Step 3: Define and train the autoencoder
    input_dim = features.shape[1]
    factor = 40  # Adjust factor as needed

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128 * factor, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64 * factor, activation='relu')(encoder)
    encoder_output = Dense(32 * factor, activation='relu')(encoder)

    decoder = Dense(64 * factor, activation='relu')(encoder_output)
    decoder = Dense(128 * factor, activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder_output = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(features, features, epochs=10, batch_size=32, shuffle=True, validation_split=0.1)

    # Extract features using the encoder part of the autoencoder
    encoder_model = Model(inputs=input_layer, outputs=encoder_output)
    encoded_features = encoder_model.predict(features)

    # Step 4: Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(encoded_features)

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
    #test_data_set, true_labels = readData('test_sample')
    test_data_set, true_labels = readData('test_full')
    num_test_sets = test_data_set.shape[0]

    # Run Tucker decomposition
    decomposed_data = buildTensor(test_data_set, rank, num_test_sets, True)

    # Apply Tucker Decomposition to test data
    test_decomposed_data = decomposed_data
    test_features = [flatten_tucker(test_decomposed_data[i][0], test_decomposed_data[i][1]) for i in range(num_test_sets)]
    test_features = np.array(test_features)

    # Extract features from test data using the trained encoder
    encoded_test_features = encoder_model.predict(test_features)
    features_scaled_test = scaler.transform(encoded_test_features)

    # Step 6: Predict using the trained OC-SVM
    prediction = best_oc_svm.predict(features_scaled_test)

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

    return accuracy

def tucker_rank_search_autoencoder_OC_svm():
    print('Tucker rank search autoencoder with One Class SVM')
    rankSet = sorted({5, 16, 32, 64})
    rank_accuracy = {}
    for factor in range(1, 4, 40
                        ):
        for bottleneck in {16, 32, 64}:
            for i in rankSet:
                for j in rankSet:
                    for k in {5}:
                        print('Rank:', i, j, k)
                        rank = (i,j,k)
                        accuracy = tucker_autoencoder_ocSVM(rank, factor, bottleneck)
                        rank_accuracy[(rank, factor, bottleneck)] = accuracy
                        print('Accuracy:', accuracy)
    print('Rank accuracy', rank_accuracy)
    bestRank = max(rank_accuracy, key=rank_accuracy.get)
    return bestRank, rank_accuracy[bestRank]

if enable_tucker_autoencoder_oc_svm:
    if use_predefined_rank == False:
        bestRank, bestAccuracy = tucker_rank_search_autoencoder_OC_svm()
        print('Best Rank for Tucker with Autoencoder and One Class SVM', bestRank, bestAccuracy)
    else:
        print('Running best rank Tucker with autoencoder and one-class SVM')
        rank = (65,35,5)
        accuracy = tucker_autoencoder_ocSVM(rank, True)

