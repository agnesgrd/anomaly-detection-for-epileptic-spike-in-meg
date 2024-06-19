import torch
import pickle
import numpy as np
import os
import os.path as op

################ SAVING & LOADING FILES, MODELS ################


def checkpoint(model, epoch, optimizer, tr_loss, val_loss, lr, filename):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'training loss': tr_loss, 'validation loss': val_loss, 'learning rate': lr}, filename)


def resume(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_obj(obj, name, path):
    with open(op.join(path + name + ".pkl"), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
    with open(op.join(path + name), "rb") as f:
        return pickle.load(f)
    
    
################ MATHS OPERATION ################

def z_score_normalize(data):
    return (data - data.mean()) / data.std()

def adjacency_matrix():
    distances = load_obj('dist_matrix.pkl', '/pbs/home/a/aguinard/DeepEpi/anomDetect/Models/')
    dist_min, dist_max = np.min(distances), np.max(distances)
    normalized_adjacency = 1 - ((distances - dist_min) / (dist_max - dist_min))
    adj_matrix = np.reshape(normalized_adjacency, (274, 274))
    return adj_matrix


################ METRICS ################
    
def get_precision(TP, FP):
    try:
        P = TP/(TP+FP)
    except ZeroDivisionError as e:
        P = 0
    return P

def get_accuracy(TP, TN, FP, FN):
    try:
        P = (TP+TN)/(TP+TN+FP+FN)
    except ZeroDivisionError as e:
        P = 0
    return P


def initialize_train_metrics():

    metrics = {
        'rec_loss': []
    }

    return metrics


def initialize_val_metrics():
    metrics = {
        'rec_loss': []
    }

    return metrics

def initialize_test_metrics():
    metrics = {

    }

    return metrics

def set_train_metrics(metrics_dict, loss):

    metrics_dict['rec_loss'].append(loss.item())

    return metrics_dict

def set_val_metrics(metrics_dict, loss):

    metrics_dict['rec_loss'].append(loss.item())

    return metrics_dict


def set_test_metrics(metrics_dict, performance):
    metrics_dict['TP'] = performance['TP'][0]
    metrics_dict['TN'] = performance['TN'][0]
    metrics_dict['FP'] = performance['FP'][0]
    metrics_dict['FN'] = performance['FN'][0]
    metrics_dict['PR'] = performance['PR'][0]
    metrics_dict['RECALL'] = performance['REC'][0]
    metrics_dict['F1'] = performance['F1'][0]

    return metrics_dict

def get_mean_metrics(metrics_dict):

    return {k: np.mean(v) for k, v in metrics_dict.items()}


################ UTILS ################

def AddValueToDict(k, d, v, i):
    # k = key - d = dict - v = value - i = type value
    # si le dictionnaire 'd' contient la clé 'k'
    # on récupère la valeur
    if k in d: i = d[k]
    # détermination du type de la valeur
    # si la valeur est de type set()
    if   isinstance(i, set):   i.add(v)
    # si la valeur est de type list()
    elif isinstance(i, list):  i.append(v)
    # si la valeur est de type str()
    elif isinstance(i, str):   i += str(v)
    # si la valeur est de type int()
    elif isinstance(i, int):   i += int(v)
    # si la valeur est de type float()
    elif isinstance(i, float): i += float(v)
    # on met à jour l'objet 'i' pour la clé 'k' dans le dictionnaire 'd'
    d[k] = i
    # on retourne le dictionnaire 'd'
    return d


################ LOG ################

class Log(object):
    """Logger class to log training metadata.

    Args:
        log_file_path (type): Log file name.
        op (type): Read or write.

    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

    Attributes:
        log (type): Description of parameter `log`.
        op
    """
    def __init__(self, log_file_path, log_file, op='r'):
        if not os.path.exists(log_file_path):
            os.mkdir(log_file_path)
        self.log = open(log_file_path+log_file, op)
        self.op = op

    def write_model(self, model):
        self.log.write('\n##MODEL START##\n')
        self.log.write(model)
        self.log.write('\n##MODEL END##\n')

        self.log.write('\n##MODEL SIZE##\n')
        self.log.write(str(sum(p.numel() for p in model.parameters())))
        self.log.write('\n##MODEL SIZE##\n')

    def log_train_metrics(self, metrics, epoch):
        self.log.write('\n##TRAIN METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')
        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##TRAIN METRICS##\n')

    def log_val_metrics(self, metrics, epoch):
        self.log.write('\n##VAL METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')
        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##VAL METRICS##\n')

    def log_test_metrics(self, metrics, epoch):
        self.log.write('\n##TEST METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')

        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##TEST METRICS##\n')

    def close(self):
        self.log.close()