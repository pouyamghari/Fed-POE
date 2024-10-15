import numpy as np
import sklearn
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def data_generator_fmnist(args):
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_labels = test_labels.reshape((test_labels.shape[0], 1))

    # Normalize pixel values to be between 0 and 1
    test_images = test_images / 255.0

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

    X_test_1, Y_test_1 = np.zeros((test_images.shape[0]//2, 28, 28, 1)), np.zeros((test_labels.shape[0]//2, 1))
    X_test_2, Y_test_2 = np.zeros((test_images.shape[0]//2, 28, 28, 1)), np.zeros((test_labels.shape[0]//2, 1))
    C = args.num_clients
    T = int(5000/C)
    maj_num = int(2000/C)
    min_num = int(500/C)
    oth_num = int(200/C)
    for i in range(C):
        X_test_1[T*i:T*i+maj_num,:,:,:] = X[int(5*i/C)][maj_num*(i%int(C/5)):maj_num*((i%int(C/5))+1),:,:,:]
        Y_test_1[T*i:T*i+maj_num,:] = Y[int(5*i/C)][maj_num*(i%int(C/5)):maj_num*((i%int(C/5))+1),:]
        cl = []
        for j in range(5):
            if j!=int(5*i/C):
                cl.append(j)
        for ind, j in enumerate(cl):
            if int(5*i/C)<j:
                X_test_1[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/5)*maj_num+min_num*i:int(C/5)*maj_num+min_num*(i+1),:,:,:]
                Y_test_1[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] =\
                Y[j][int(C/5)*maj_num+min_num*i:int(C/5)*maj_num+min_num*(i+1),:]
            else:
                X_test_1[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/5)*maj_num+min_num*(i-int(C/5)):int(C/5)*maj_num+min_num*(i-int(C/5)+1),:,:,:]
                Y_test_1[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] = \
                Y[j][int(C/5)*maj_num+min_num*(i-int(C/5)):int(C/5)*maj_num+min_num*(i-int(C/5)+1),:]
        for j in range(5,10):
            X_test_1[T*i+maj_num+4*min_num+(j-5)*oth_num:T*i+maj_num+4*min_num+(j-4)*oth_num,:,:,:] = \
            X[j][int(C/5)*maj_num+4*int(C/5)*min_num+i*oth_num:int(C/5)*maj_num+4*int(C/5)*min_num+(i+1)*oth_num,:,:,:]
            Y_test_1[T*i+maj_num+4*min_num+(j-5)*oth_num:T*i+maj_num+4*min_num+(j-4)*oth_num,:] = \
            Y[j][int(C/5)*maj_num+4*int(C/5)*min_num+i*oth_num:int(C/5)*maj_num+4*int(C/5)*min_num+(i+1)*oth_num,:]

    for i in range(C):
        X_test_1[T*i:T*(i+1),:,:,:], Y_test_1[T*i:T*(i+1),:] = shuffle(X_test_1[T*i:T*(i+1),:,:,:], Y_test_1[T*i:T*(i+1),:])

    for i in range(C):
        X_test_2[T*i:T*i+maj_num,:,:,:] = X[5+int(5*i/C)][maj_num*(i%int(C/5)):maj_num*((i%int(C/5))+1),:,:,:]
        Y_test_2[T*i:T*i+maj_num,:] = Y[5+int(5*i/C)][maj_num*(i%int(C/5)):maj_num*((i%int(C/5))+1),:]
        cl = []
        for j in range(5,10):
            if j!=5+int(5*i/C):
                cl.append(j)
        for ind, j in enumerate(cl):
            if 5+int(5*i/C)<j:
                X_test_2[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/5)*maj_num+min_num*i:int(C/5)*maj_num+min_num*(i+1),:,:,:]
                Y_test_2[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] =\
                Y[j][int(C/5)*maj_num+min_num*i:int(C/5)*maj_num+min_num*(i+1),:]
            else:
                X_test_2[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:,:,:] = \
                X[j][int(C/5)*maj_num+min_num*(i-int(C/5)):int(C/5)*maj_num+min_num*(i-int(C/5)+1),:,:,:]
                Y_test_2[T*i+maj_num+min_num*ind:T*i+maj_num+min_num*(ind+1),:] = \
                Y[j][int(C/5)*maj_num+min_num*(i-int(C/5)):int(C/5)*maj_num+min_num*(i-int(C/5)+1),:]
        for j in range(5):
            X_test_2[T*i+maj_num+4*min_num+j*oth_num:T*i+maj_num+4*min_num+(j+1)*oth_num,:,:,:] = \
            X[j][int(C/5)*maj_num+4*int(C/5)*min_num+i*oth_num:int(C/5)*maj_num+4*int(C/5)*min_num+(i+1)*oth_num,:,:,:]
            Y_test_2[T*i+maj_num+4*min_num+j*oth_num:T*i+maj_num+4*min_num+(j+1)*oth_num,:] = \
            Y[j][int(C/5)*maj_num+4*int(C/5)*min_num+i*oth_num:int(C/5)*maj_num+4*int(C/5)*min_num+(i+1)*oth_num,:]

    for i in range(C):
        X_test_2[T*i:T*(i+1),:,:,:], Y_test_2[T*i:T*(i+1),:] = shuffle(X_test_2[T*i:T*(i+1),:,:,:], Y_test_2[T*i:T*(i+1),:])

    X_test, Y_test = np.append(X_test_1, X_test_2, axis=0), np.append(Y_test_1, Y_test_2, axis=0)

    return X_test, Y_test