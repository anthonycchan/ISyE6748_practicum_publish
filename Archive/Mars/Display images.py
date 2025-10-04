import numpy as np
from matplotlib import pyplot as plt
import os
import tensorly.decomposition
import tensorly as tl
from sklearn.svm import OneClassSVM

'''
#img_array = np.load('test_typical/mcam08560_MR_1_sol1652_11.npy')
img_array = np.load('train_typical/mcam00006_MR_0_sol0003_4.npy')
#img_array = np.load('test_novel/all/mcam00844_R0_sol0150_5.npy')

print(img_array.shape, img_array.shape[2], type(img_array))

for i in range(img_array.shape[2]):
    print(img_array[:,:,i].shape, i)
    plt.imshow(img_array[:,:,i], cmap='gray')
    plt.show()
'''


directory = os.fsencode('train_sample')

print("directory", os.fsdecode(directory))
filelist = os.listdir(directory)
numFiles = len(filelist)
print("num files:", numFiles)
X = np.ones([numFiles, 64, 64, 6])
print("X:", X.shape)
i=0
for file in filelist:
    filename = os.fsdecode(file)
    print(filename)
    img_array = np.load(os.fsdecode(directory) + "/" + filename)
    print(img_array.shape, type(img_array))
    X[i,:,:,:] = img_array
    i+=1

P = tl.decomposition.parafac(X, 5, verbose=False)
X_estimated = tl.kruskal_to_tensor(P)  # Reconstruct
print("X_estimated:", X_estimated.shape)

'''
for i in range(0,numFiles):
    plt.imshow(X_estimated[i,:,:,1], cmap='gray')
    plt.show()
'''

k,x,y,z=X_estimated.shape
matricizedX = X_estimated.reshape((k, x * y * z ))
print("matricizedX", matricizedX.shape)

# One-class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
oc_svm.fit(matricizedX)

y_pred_train = oc_svm.predict(matricizedX)
print("Train predictions:", y_pred_train)