# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch.optim as optim

# Data Manipulation
import cupy as cp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pandas as pd

# Sklearn

# Local Imports
from utils import AddValueToDict

# Other
import params
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#########################ERROR CARDS##########################       
def get_error_cards_cupy(model, dataloader, device):
    model.eval()
    print('here')

    with torch.no_grad():

        list_X = list()
        list_Y = list()

        # get predictions
        for data in dataloader:
            inputs, targets = data

            inputs = inputs.to(device)
            inputs = inputs.float()
            
            outputs = model(inputs)
            
            inputs = inputs.squeeze(dim=1)
            outputs = outputs.squeeze(dim=1)

            targets = targets.to(device)
            targets = targets.squeeze(dim=1)

            inputs = to_dlpack(inputs)
            targets = to_dlpack(targets)
            outputs = to_dlpack(outputs)

            X = cp.square(cp.from_dlpack(outputs) - cp.from_dlpack(inputs))
            Y = cp.from_dlpack(targets)

            list_X.append(X)
            list_Y.append(Y)

        X = cp.concatenate([x for x in list_X])
        X = cp.swapaxes(X, 1, 2)
        X = cp.vstack(X)

        Y = cp.concatenate([y for y in list_Y])
        Y = cp.hstack(Y)

        # X = cp.concatenate([x for x in list_X])
        # dims = X.shape
        # X = X.reshape(dims[0], dims[1]*dims[2])
        # Y = cp.concatenate([y for y in list_Y])

        return X, Y

def get_error_cards_cpu(model, dataloader, device, reduce_features):
    model.eval()

    with torch.no_grad():

        list_X = list()
        list_Y = list()

        # get predictions
        for data in dataloader:
            inputs, targets = data

            inputs = inputs.to(device)
            inputs = inputs.float()
            
            outputs = model(inputs)
            
            inputs = inputs.squeeze(dim = 1)
            outputs = outputs.squeeze(dim = 1)
            targets = targets.squeeze(dim = 1)

            inputs = inputs.to("cpu")
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")

            X = np.square(np.array(outputs) - np.array(inputs))
            Y = np.array(targets)

            list_X.append(X)
            list_Y.append(Y)

        X = np.concatenate([x for x in list_X])
        X = np.swapaxes(X, 1, 2)
        X = np.vstack(X)

        Y = np.concatenate([y for y in list_Y])
        Y = np.hstack(Y)
        
        if reduce_features:
            X = X[:,::10]
        return X, Y
    
def get_error_cards_gpu(model, dataloader, device, reduce_features):
    model.eval()

    with torch.no_grad():

        list_X = list()
        list_Y = list()

        # get predictions
        for data in dataloader:
            inputs, targets = data

            inputs = inputs.to(device)
            inputs = inputs.float()
            
            outputs = model(inputs)
            
            inputs = inputs.squeeze(dim = 1)
            outputs = outputs.squeeze(dim = 1)
            targets = targets.squeeze(dim = 1)

            inputs = inputs.to("cpu")
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")

            X = np.square(np.array(outputs) - np.array(inputs))
            Y = np.array(targets)

            list_X.append(X)
            list_Y.append(Y)

            list_X.append(X)
            list_Y.append(Y)

        X = np.concatenate([x for x in list_X])
        X = np.swapaxes(X, 1, 2)
        X = np.vstack(X)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=0)

        Y = np.concatenate([y for y in list_Y])
        Y = np.hstack(Y)

        return torch.tensor(X, dtype=torch.float32).to("cuda:0"), torch.tensor(Y, dtype=torch.float32).to("cuda:0")

def get_error_cards_mega_old(model, dataloader, device):
    model.eval()
    with torch.no_grad():

        y_pred = list()
        y_true = list()

        # get predictions
        for data in dataloader:
            input, target = data

            input = input.to(device)
            idx = torch.arange(274).to(device)

            output = model(input, idx, device)

            criterion = nn.MSELoss(reduction='none')

            score = criterion(output[0][0], output[0][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            + criterion(output[1][0], output[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            + criterion(output[2][0], output[2][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            + criterion(output[3][0], output[3][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            + criterion(output[4][0], output[4][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)

            y_pred.append(score.cpu().numpy().reshape(-1))
            y_true.append(torch.max(target, 1).values.cpu().numpy().reshape(-1))


    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    return y_pred, y_true

def get_error_cards_mega_nn(model, dataloader, device, level):

    model.eval()
    all_score = []
    labels = []
    with torch.no_grad():
        
        # get predictions
        for data in tqdm(dataloader):
            score = []
            score_spike = []
            score_normal = []

            input, target = data

            input = input.to(device)
            target = target.to(device)
            idx = torch.arange(274).to(device)

            output = model(input, idx, device)

            criterion = nn.MSELoss(reduction='none')
            
            # criterion = [batch_size * window_size/2/2/./2 * channels]
            for i in level:
                c_i = criterion(output[i][0], output[i][1])
                c_i = c_i.permute(0, 2, 1)
                c_i = F.interpolate(c_i, size = params.sfreq*params.window_size, mode = 'nearest')
                c_i = c_i.permute(0, 2, 1) #[batch_size * window_size * channels]
                score.append(c_i)
            
            # anomaly score and labels post_processing
            score = torch.stack(score) #[criterion * batch_size * window_size * channels]
            score = score[:,:,45:255,:]
            print(score.mean(dim=1, keepdim=True).shape)
            score = (score - score.mean(dim=1, keepdim=True))/score.std(dim=1, keepdim = True)
            score = score.permute(1, 2, 3, 0) #[batch_size * window_size * channels * criterion]
            target = target[:,45:255]
            spike_loc = torch.argwhere(target==1).unsqueeze(2).unsqueeze(3)
            print(spike_loc)




            all_score.append(score)
            labels.append(target[:,45:255])
        
        X = torch.vstack(all_score) #[all_batch * window_size * channels * criterion]
        X = torch.tensor_split(X, 9, dim = 1)
        X = torch.vstack(X) #[all_batch*10 * window_size/10 * channels * criterion]
        X = X.permute(0, 3, 1, 2) #[all_batch*10 *criterion * window_size/10 * channels]

        Y = torch.vstack(labels) #[all_batch * window_size]
        Y = torch.flatten(Y).reshape(X.shape[0], X.shape[2])
        Y = torch.max(Y, dim = 1).values #[all_batch*10 --> max(window_size/10)]
        Y = Y.unsqueeze(1)
        print(Y.shape[0])
        # Y = F.one_hot(Y.unsqueeze(1).long(), num_classes = 2)

        return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)
    
def get_raw_error_cards_mega(model, dataloader, device):

    model.eval()
    all_score = []
    labels = []
    with torch.no_grad():
        
        # get predictions
        for data in tqdm(dataloader):
            score = []

            input, target = data

            input = input.to(device)
            target = target.to(device)
            idx = torch.arange(274).to(device)

            output = model(input, idx, device)

            criterion = nn.MSELoss(reduction='none')
            
            # criterion = [batch_size * window_size/2/2/./2 * channels]
            for i in range(5):
                c_i = criterion(output[i][0], output[i][1])
                c_i = c_i.permute(0, 2, 1)
                c_i = F.interpolate(c_i, size = params.sfreq*params.window_size, mode = 'nearest')
                c_i = c_i.permute(0, 2, 1) #[batch_size * window_size * channels]
                score.append(c_i)
            
            score = torch.stack(score) #[criterion * batch_size * window_size * channels]
            score = score.permute(1, 2, 3, 0) #[batch_size * window_size * channels * criterion]
            all_score.append(score)
            labels.append(target)
        
        X = torch.vstack(all_score) #[all_batch * window_size * channels * criterion]
        X = torch.reshape(X, (-1, 274, 5))

        # new_criterion = X[:,:,2]-0.5*(X[:,:,1]+X[:,:,3])

        # X = torch.cat((X, new_criterion.unsqueeze(2)), dim=2)

        Y = torch.vstack(labels) #[all_batch * window_size]
        Y = torch.flatten(Y) 

        print(f"{X.shape[0]} windows with {X.shape[1]} features of length 1 timesteps x mean of 274 channels")
        print(f"Number of windows with spike: {(Y==1).sum()} / without: {(Y==0).sum()}")
        print(f"Ratio: {(Y==1).sum()/Y.shape[0]*100:.2f}%")

        return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)

def old_get_error_cards_mega_unfolded(model, dataloader, device):

    model.eval()
    score = {}
    labels = []
    with torch.no_grad():
        # get predictions
                
        for data in dataloader:
            input, target = data

            input = input.to(device)
            idx = torch.arange(274).to(device)

            output = model(input, idx, device)

            criterion = nn.MSELoss(reduction='none')

            for i in range(5):
                c = criterion(output[i][0], output[i][1]).cpu().numpy()
                t = np.linspace(0, 1, c.shape[1])
                new_t = np.linspace(0, 1, params.sfreq*params.window_size)
                f_interp = interp1d(t, c, axis=1)
                new_c = f_interp(new_t)
                score = AddValueToDict(f'{i}', score, new_c, list())
            labels.append(target)


        for key, value in score.items():
            score[key] = np.vstack(value)
            score[key] = np.swapaxes(score[key], 0, 1)
            score[key] = np.vstack(score[key])
            # score[key] = np.mean(score[key], axis =-1)
    
        labels = np.vstack(labels)
        labels = np.swapaxes(labels, 0, 1)
        labels = np.hstack(labels)

        score_arr = np.stack(list(score.values()))

        # score_df = pd.DataFrame.from_dict(score)
        X = torch.tensor(score_arr)
        X = torch.swapaxes(X, 0, 1)
        Y = torch.tensor(labels)

        X = X.unfold(0, 30, 10)
        X = torch.swapaxes(X, 2, 3)

        Y = Y.unfold(0, 30, 10)
        Y = torch.any(Y==1, axis=1).to(int)
        print(np.count_nonzero(Y==1))
        #Y = Y.unsqueeze(1)
        Y = F.one_hot(Y.unsqueeze(1), num_classes = 2)

        print("Shape of X:", X.shape)
        print("Shape of labels (one-hot encoding):", Y.shape)

        return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)

def get_error_cards_mega_rolling_windows(model, dataloader, device):
    model.eval()
    score = {}
    labels = []

    with torch.no_grad():
        # get predictions
               
            for data in dataloader:
                input, target = data

                input = input.to(device)
                idx = torch.arange(274).to(device)

                output = model(input, idx, device)

                criterion = nn.MSELoss(reduction='none')

                for i in range(5):
                    c = criterion(output[i][0], output[i][1]).cpu().numpy()
                    t = np.linspace(0, 1, c.shape[1])
                    new_t = np.linspace(0, 1, params.sfreq*params.window_size)
                    f_interp = interp1d(t, c, axis=1)
                    new_c = f_interp(new_t)
                    score = AddValueToDict(f'{i}', score, new_c, list())
                labels.append(target)


            for key, value in score.items():
                score[key] = np.vstack(value)
                score[key] = np.swapaxes(score[key], 0, 1)
                score[key] = np.vstack(score[key])
                score[key] = np.mean(score[key], axis =-1)
        
            labels = np.vstack(labels)
            labels = np.swapaxes(labels, 0, 1)
            labels = np.hstack(labels)

            score_df = pd.DataFrame.from_dict(score)
            rolling_score = score_df.rolling(window=3, center=True, step = 3).mean()
            rolling_score = rolling_score.dropna()
            rolling_labels = pd.DataFrame(labels, columns=['labels'])
            rolling_labels['labels'] = rolling_labels.rolling(window = 5, center=True, step = 3).max()
            rolling_labels = rolling_labels.dropna().astype(int)

            return rolling_score, rolling_labels

def get_latent_features_mega(model, dataloader, device):
    model.eval()
    latent1_list = list()
    latent2_list = list()
    latent3_list = list()
    labels = list()

    with torch.no_grad():

        # get predictions
        for i,data in tqdm(enumerate(dataloader)):
            if i % 4 == 0:
                input, target = data

                input = input.to(device)
                target = target.to(device)
                idx = torch.arange(274).to(device)

                output = model(input, idx, device, latent = True) # [batch_size x filter_size x 274 channels x window size/2/./2]
                
                latent1, latent2, latent3 = output

                del output

                new_lat = list()
                for lat in [latent1, latent2, latent3]:
                    lat = lat.permute(0,1,3,2) #[batch_size x filter_size x 1 x window_size/2/./2 x channels]
                    lat = F.interpolate(lat, size = (params.sfreq*params.window_size, 274), mode = 'nearest')
                    lat = lat.permute(0,2,3,1) #[batch_size x window_size/2/./2 x channels x filter_size]
                    new_lat.append(lat)

                latent1, latent2, latent3 = new_lat
                    
                latent1_list.append(latent1)
                latent2_list.append(latent2)
                latent3_list.append(latent3)
                labels.append(target)

        latent1 = torch.vstack(latent1_list) #[all_batch x window_size x channels x filters]
        latent1 = torch.reshape(latent1, (-1, 274, params.filters[0])) #[each timesteps of all batches x channels x filter_size]
        latent1 = torch.reshape(latent1, (latent1.shape[0], 274*params.filters[0])) #[each timesteps of all batches x channels*filter_size]

        latent2 = torch.vstack(latent2_list)
        latent2 = torch.reshape(latent2, (-1, 274, params.filters[0]))
        latent2 = torch.reshape(latent2, (latent2.shape[0], 274*params.filters[0]))

        latent3 = torch.vstack(latent3_list)
        latent3 = torch.reshape(latent3, (-1, 274, params.filters[0]))
        latent3 = torch.reshape(latent3, (latent2.shape[0], 274*params.filters[0]))

        Y = torch.vstack(labels)
        Y = torch.flatten(Y)

        return (latent1, latent2, latent3), Y
