import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from lib.models.model_loader import model_loader
from lib.datasets.data_loader import data_loader
from lib.FedPOE.get_FedPOE import get_FedPOE
from lib.FedPOE.get_predictor import get_predictor

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='CIFAR-10', type=str)
parser.add_argument("--task", default='classification', type=str)
parser.add_argument("--num_clients", default=20, type=int)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--period", default=20, type=int)
parser.add_argument("--num_models", default=10, type=int)

args = parser.parse_args()

X, Y = data_loader(args)

T = X[0].shape[0] #T: Time Horizon
args.eta = .01/np.sqrt(T)
args.eta_c = 1/np.sqrt(T)

acc = np.zeros((args.num_clients, T))

model = model_loader(args)

local, federated = get_FedPOE(model, args)

predictors = get_predictor(args)

for t in range(T):
    aggregated_models = []
    for i in range(args.num_clients):
        indices = predictors[i].model_selection()
        models = [local[i].model, federated.model]
        for ind in indices:
            models.append(local[i].dic[ind])
        X_new, Y_new = X[i][t:t+1,:,:,:], Y[i][t:t+1,:]
        prediction = predictors[i].prediction(X_new, Y_new, models, indices, t)
        if prediction==Y_new[0,0]:
            acc[i,t] = 1
        if t>args.batch_size:
            X_update, Y_update = predictors[i].X[-args.batch_size:,:,:,:], predictors[i].Y[-args.batch_size:,:]
        else:
            X_update, Y_update = predictors[i].X, predictors[i].Y
        aggregated_models.append(federated.client_update(X_update, Y_update, 1))
        local[i].update([local[i].client_update(X_update, Y_update, 1)], t)
    federated.update(aggregated_models, t)

print('Accuracy of Fed-POE is %s' %np.mean(np.mean(acc, axis=1)))
print('Standard deviation of Fed-POE is %s' %np.std(np.mean(acc, axis=1)))