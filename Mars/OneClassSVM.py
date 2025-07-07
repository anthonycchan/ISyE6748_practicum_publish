from sklearn.svm import OneClassSVM
import numpy as np

# Generate some training data (normal data)
X_train = np.random.normal(0, 1, (100, 2))

# Generate some test data (contains normal and anomalous data)
X_test = np.concatenate([np.random.normal(0, 1, (20, 2)), np.random.uniform(5, 6, (5, 2))])

# Fit the OC-SVM model
oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
oc_svm.fit(X_train)

# Predict anomalies
y_pred_train = oc_svm.predict(X_train)  # Should be mostly 1's
y_pred_test = oc_svm.predict(X_test)    # Should contain -1's for anomalies

print("Train predictions:", y_pred_train)
print("Test predictions:", y_pred_test)
