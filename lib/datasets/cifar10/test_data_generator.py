import numpy as np
import sklearn
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def data_generator_cifar10(args):
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    X, Y = {}, {}
    num_labels = np.zeros((1,10))

    for i in range(0,np.prod(test_labels.shape)):
        for j in range(10):
            if test_labels[i,0]==j and num_labels[0,j]==0:
                X[j], Y[j] = test_images[i:i+1,:,:,:], test_labels[i:i+1,:,]
                num_labels[0,j]+=1
            elif test_labels[i,0]==j and num_labels[0,j]>0:
                X[j], Y[j] = np.append(X[j], test_images[i:i+1,:,:,:], axis=0), np.append(Y[j], test_labels[i:i+1,:,], axis=0)
                num_labels[0,j]+=1

    X_test, Y_test = np.zeros(test_images.shape), np.zeros(test_labels.shape)
    C = args.num_clients
    T = int(10000/C)
    maj_num = int(5500/C)
    min_num = int(500/C)
    for i in range(C):
        X_test[T*i:T*i+maj_num,:,:,:] = X[int(10*i/C)][maj_num*(i%int(C/10)):maj_num*((i%int(C/10))+1),:,:,:]
        Y_test[T*i:T*i+maj_num,:] = Y[int(10*i/C)][maj_num*(i%int(C/10)):maj_num*((i%int(C/10))+1),:]
        cl = []
        for j in range(10):
            if j!=int(10*i/C):
                cl.append(j)
        for ind, j in enumerate(cl):
            if int(10*i/C)<j:
                X_test[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/10)*maj_num+min_num*i:int(C/10)*maj_num+min_num*(i+1),:,:,:]
                Y_test[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] =\
                Y[j][int(C/10)*maj_num+min_num*i:int(C/10)*maj_num+min_num*(i+1),:]
            else:
                X_test[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/10)*maj_num+min_num*(i-int(C/10)):int(C/10)*maj_num+min_num*(i-int(C/10)+1),:,:,:]
                Y_test[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] = \
                Y[j][int(C/10)*maj_num+min_num*(i-int(C/10)):int(C/10)*maj_num+min_num*(i-int(C/10)+1),:]

    for i in range(C):
        X_test[T*i:T*(i+1),:,:,:], Y_test[T*i:T*(i+1),:] = shuffle(X_test[T*i:T*(i+1),:,:,:], Y_test[T*i:T*(i+1),:])

    return X_test, Y_test