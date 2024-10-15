import numpy as np
from lib.datasets.cifar10.test_data_generator import data_generator_cifar10
from lib.datasets.fmnist.test_data_generator import data_generator_fmnist
from lib.datasets.air.data_processor import data_processing_air
from lib.datasets.wec.data_processor import data_processing_wec

def data_loader(args):
    if args.dataset=="CIFAR-10":
        samples, labels = data_generator_cifar10(args)
        
        total_num_samples = samples.shape[0]
        T = int(np.floor(total_num_samples/args.num_clients))
        
        X = []
        Y = []
        
        for i in range(args.num_clients):
            X.append(samples[i*T:i*T+1,:,:,:])
            Y.append(labels[i*T:i*T+1,:])
        
        for t in range(T):
            for i in range(args.num_clients):
                X[i], Y[i] = np.append(X[i], samples[t+i*T:t+i*T+1,:,:,:], axis=0), np.append(Y[i], labels[t+i*T:t+i*T+1,:], axis=0)
    
    elif args.dataset=="FMNIST":
        samples, labels = data_generator_fmnist(args)
        
        total_num_samples = samples.shape[0]
        T = int(np.floor(total_num_samples/args.num_clients))
        
        X = []
        Y = []
        
        for i in range(args.num_clients):
            X.append(samples[i*T:i*T+1,:,:,:])
            Y.append(labels[i*T:i*T+1,:])
        
        for t in range(T):
            for i in range(args.num_clients):
                X[i], Y[i] = np.append(X[i], samples[t+i*T:t+i*T+1,:,:,:], axis=0), np.append(Y[i], labels[t+i*T:t+i*T+1,:], axis=0)
    
    elif args.dataset=="Air":
        X, Y = data_processing_air(args)
    
    elif args.dataset=="WEC":
        X, Y = data_processing_wec(args)
    
    return X, Y