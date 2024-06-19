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
from sklearn.preprocessing import MinMaxScaler

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Local Imports
from Models import clf_nn
import inference as infer
import loop as loop
from utils import resume, get_accuracy, set_test_metrics, initialize_test_metrics, checkpoint
from utils import AddValueToDict
from saving_functions import plot_pca, plot_lda
import saving_functions as save

# Other
from tqdm.std import tqdm
from threading import Thread
from statistics import mean
import params
import warnings
warnings.filterwarnings('ignore')

######################### CLASSICA L##########################       
def classifier_xgboost(model, train_dataloader, test_dataloader, device):

    X_train, Y_train = infer.get_error_cards_cupy(model, train_dataloader, device)
    X_test, Y_test = infer.get_error_cards_cupy(model, test_dataloader, device)

    # Create an XGBoost classifier
    xgb_model = XGBClassifier(tree_method='hist', device=device, scale_pos_weight=35)

    # Train the classifier
    xgb_model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = xgb_model.predict(X_test)

    return Y_test.get(), Y_pred

def classifier_svm(model, train_dataloader, test_dataloader, device):

    X_train, Y_train = infer.get_error_cards_cpu(model, train_dataloader, device, reduce_features = True)
    X_test, Y_test = infer.get_error_cards_cpu(model, test_dataloader, device, reduce_features = True)

    # Create a classifier
    clf = SVC(class_weight = "balanced")

    # Train the classifier
    clf.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = clf.predict(X_test)

    return Y_test, Y_pred

def classifier_nn(model, train_dataloader, test_dataloader, simulation_nb, device):

    X_train, Y_train = infer.get_error_cards_gpu(model, train_dataloader, device, reduce_features = False)
    X_test, Y_test = infer.get_error_cards_gpu(model, test_dataloader, device, reduce_features = False)

    train_dataloader_clf = DataLoader(X_train, Y_train, batch_size = 64, shuffle = True)
    test_dataloader_clf = DataLoader(X_test, Y_test, batch_size = 64, shuffle = True)

    # Training loop
    loss_valid_all = []
    loss_train_all = []
    f1_score_all = []

    print("Starting classifier training")

    num_epochs = 10
    optimizer = "Adam"
    criterion = nn.BCEWithLogitsLoss()
    model_name = 'SFCN'
    clf_model = clf_nn.SFCN().to(device)
    criterion.to(device)

    print(summary(clf_model, (1, 274, 30)))

    for epoch in range(num_epochs):
        print("Epoch:", epoch, "/", num_epochs)

        ##############Training###################
        print(X_train.shape)
        print(Y_train.shape)
        training_loss = loop.train_epoch(
            clf_model, train_dataloader_clf, optimizer, criterion, device
        )

        print("Training loss:", training_loss)

        ##############Validation################       

        validation_loss = loop.validate_epoch(clf_model, test_dataloader_clf, criterion, device)

        print("Validation loss:", validation_loss)

        loss_train_all.append(training_loss)
        loss_valid_all.append(validation_loss)

    resume(model, params.path_writing_data + "best_" + model_name + "_" + str(simulation_nb) + ".pth")

    return clf_model

######################### MEGAAAAA ##########################       
def adjust_predicts(score, label, data_category=None):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    new_array = np.copy(score)
    label = np.asarray(label)
    actual = label == 1
    anomaly_count = 0
    max_score = 0.0
    for i in tqdm(range(len(score))):
        if actual[i]:
            max_score = new_array[i]
            anomaly_count += 1
            for j in range(i - 1, -1, -1):
                if not actual[j]:
                    new_array[j + 1:i + 1] = max_score
                    break
                else:
                    if new_array[j] > max_score:
                        max_score = new_array[j]
    return new_array

def calc_point2point(predict, actual):
    """Calculate f1 score by predict and actual.

    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN

def bf_search(y_pred, y_true):
    print('Using brute force search for the proper threshold...')

    candidate_values = np.concatenate(
        [y_pred[y_true == 0], np.sort(y_pred[y_true == 1])[:y_pred[y_true == 1].shape[0] // 5]], axis=0)
    candidates = np.linspace(np.min(y_pred[y_true == 0]), np.max(candidate_values), 10000)

    f1s = np.zeros_like(candidates)
    precisions = np.zeros_like(candidates)
    recalls = np.zeros_like(candidates)
    tps = np.zeros_like(candidates)
    tns = np.zeros_like(candidates)
    fps = np.zeros_like(candidates)
    fns = np.zeros_like(candidates)

    y_pred = adjust_predicts(y_pred, y_true)

    def calc_metric(th, num):
        y_res = np.zeros_like(y_pred)
        y_res[y_pred >= th] = 1.0

        p_t = calc_point2point(y_res, y_true)

        f1s[num] = p_t[0]
        precisions[num] = p_t[1]
        recalls[num] = p_t[2]
        tps[num] = p_t[3]
        tns[num] = p_t[4]
        fps[num] = p_t[5]
        fns[num] = p_t[6]

    tasks = []
    for i in tqdm(range(len(candidates))):
        th = Thread(target=calc_metric, args=(candidates[i], i))
        th.start()
        tasks.append(th)

    for th in tasks:
        th.join()

    best_f1_ind = np.argmax(tps)
    performance = {'F1': f1s[best_f1_ind], 'PR': precisions[best_f1_ind], 'REC': recalls[best_f1_ind],
                   'TP': tps[best_f1_ind], 'TN': tns[best_f1_ind], 'FP': fps[best_f1_ind], 'FN': fns[best_f1_ind]}
    performance = pd.DataFrame(performance, index=[0])

    return performance

def classifier_mega_xgboost(model, train_dataloader, test_dataloader, criterion, simulation_nb, device):
    
    X_train, Y_train = infer.get_error_cards_mega_rolling_windows(model, train_dataloader, device)
    X_test, Y_test = infer.get_error_cards_mega_rolling_windows(model, test_dataloader, device)

    # Create an XGBoost classifier
    xgb_model = XGBClassifier(eval_metric = "auc", 
                                learning_rate = 0.01, 
                                n_estimators=1000,
                                max_depth=10,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective= 'binary:logistic',
                                nthread=4,
                                scale_pos_weight=35, 
                                device = device)

    # Train the classifier
    xgb_model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = xgb_model.predict(X_test)

    return Y_test, Y_pred

def classifier_mega_svm(model, train_dataloader, test_dataloader, criterion, simulation_nb, device):
    
    X_train, Y_train = infer.get_error_cards_mega_rolling_windows(model, train_dataloader, device)
    X_test, Y_test = infer.get_error_cards_mega_rolling_windows(model, test_dataloader, device)

    # Create a classifier
    clf = SVC() #class_weight = "balanced")

    # Train the classifier
    clf.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = clf.predict(X_test)

    return Y_test, Y_pred

def classifier_mega_nn(model, train_dataloader, test_dataloader, simulation_nb, device):
    
    batch_size = 240

    X_train, Y_train = infer.get_error_cards_mega_nn(model, train_dataloader, device, level = [3])
    X_test, Y_test = infer.get_error_cards_mega_nn(model, test_dataloader, device, level = [3])
    
    print("----------------------------------------------------")
    print("CLASSIFIER DATASET")

    print(f"Training with {X_train.shape[0]} windows with {X_train.shape[1]} features (frequency-band) of length {X_train.shape[2]} timesteps x {X_train.shape[3]} channels")
    print(f"Number of windows with spike: {(Y_train==1).sum(dim=0).squeeze()} / without: {(Y_train==0).sum(dim=0).squeeze()}")
    print(f"Ratio: {(Y_train==1).sum(dim=0).squeeze()/Y_train.shape[0]*100:.2f}%")

    print(f"Testing with {X_test.shape[0]} windows with {X_test.shape[1]} features (frequency-band) of length {X_test.shape[2]} timesteps x {X_test.shape[3]} channels")
    print(f"Number of windows with spike: {(Y_test==1).sum(dim=0).squeeze()} / without: {(Y_test==0).sum(dim=0).squeeze()}")
    print(f"Ratio: {(Y_test==1).sum(dim=0).squeeze()/Y_test.shape[0]*100:.2f}%")

    # weight = [1./(Y_train==0).sum(dim=0).squeeze(),1./(Y_train==1).sum(dim=0).squeeze()]
    # samples_weight = torch.tensor([weight[t] for t in [0,1]]).double()
    # print(samples_weight)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    trainset = torch.utils.data.TensorDataset(X_train, Y_train)
    testset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_dataloader_clf = DataLoader(trainset, batch_size = batch_size)
    test_dataloader_clf = DataLoader(testset, batch_size= batch_size, shuffle=True)

    # Training loop
    loss_valid_all = []
    loss_train_all = []
    f1_score_all = []
    lr_all = []

    print("----------------------------------------------------")
    print("STARTING CLASSIFIER TRAINING")
    print(f"Batch size = {batch_size}")
    num_epochs = 80
    optimizer = "Adam"
    pos_weight = torch.tensor([35.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model_name = 'SFCN_mega'
    clf_model = clf_nn.SFCN_mega().to(device)
    optimizer = optim.Adam(
        clf_model.parameters(),
        lr=0.0001,
        weight_decay=params.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=params.gammaLR)
    criterion.to(device)

    print(summary(clf_model, (1, 30, 274)))

    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1, "/", num_epochs)

        ##############Training###################
        training_loss = loop.train_epoch(
            clf_model, train_dataloader_clf, optimizer, criterion, device
        )

        print("Training loss:", training_loss)

        spike_gt, spike_pred = loop.test_mega_epoch(clf_model, train_dataloader_clf, criterion, device)

        accuracy = accuracy_score(spike_gt, spike_pred)
        cm = confusion_matrix(spike_gt, spike_pred)
        print("Confusion matrix", cm)
        report = classification_report(spike_gt, spike_pred, target_names=["no spike", "w spike"], labels=[0,1], output_dict=True)
        f1_score = report['w spike']['f1-score']

        ##############Validation################       
        validation_loss = loop.validate_epoch(clf_model, test_dataloader_clf, criterion, device)

        print("Validation loss:", validation_loss)

        spike_gt, spike_pred = loop.test_mega_epoch(clf_model, test_dataloader_clf, criterion, device, test = True)

        accuracy = accuracy_score(spike_gt, spike_pred)
        cm = confusion_matrix(spike_gt, spike_pred)
        print("Confusion matrix", cm)
        report = classification_report(spike_gt, spike_pred, target_names=["no spike", "w spike"], labels=[0,1], output_dict=True)
        f1_score = report['w spike']['f1-score']

        loss_train_all.append(training_loss)
        loss_valid_all.append(validation_loss)
        f1_score_all.append(f1_score)
        lr_all.append(scheduler.get_lr()[0])

        save.plot_epochs_metric(
    loss_train_all, loss_valid_all, f1_score_all, params.path_writing_data, model_name + "_" + str(simulation_nb)
)

    checkpoint(clf_model, epoch, optimizer, loss_train_all, loss_valid_all, lr_all, params.path_writing_data + "best_" + model_name + "_" + str(simulation_nb) + ".pth")

    save.plot_epochs_metric(
    loss_train_all, loss_valid_all, f1_score_all, params.path_writing_data, model_name + "_" + str(simulation_nb)
)

    return spike_gt, spike_pred   

def classifier_mega_xgboost_latent(model, train_dataloader, test_dataloader, device):
    
    print("----------------------------------------------------")
    print("CLASSIFIER DATASET")

    X_train, Y_train = infer.get_latent_features_mega(model, train_dataloader, device)
    X_test, Y_test = infer.get_latent_features_mega(model, test_dataloader, device)

    X_train = cp.asarray(torch.cat(X_train, dim=1))
    # X_train = cp.asarray(torch.mean(torch.cat(X_train, dim=2), 1))
    Y_train = cp.asarray(Y_train)
    X_test = cp.asarray(torch.cat(X_test, dim=1))
    # X_test = cp.asarray(torch.mean(torch.cat(X_test, dim=2), 1))
    Y_test = cp.asarray(Y_test)

    print(f"Training with {X_train.shape[0]} windows with {X_train.shape[1]} features of length 1 timesteps x mean of 274 channels")
    print(f"Number of windows with spike: {(Y_train==1).sum()} / without: {(Y_train==0).sum()}")
    print(f"Ratio: {(Y_train==1).sum()/Y_train.shape[0]*100:.2f}%")

    print(f"Testing with {X_test.shape[0]} windows with {X_test.shape[1]} features of length 1 timesteps x mean of 274 channels")
    print(f"Number of windows with spike: {(Y_test==1).sum()} / without: {(Y_test==0).sum()}")
    print(f"Ratio: {(Y_test==1).sum()/Y_test.shape[0]*100:.2f}%")


    # Create an XGBoost classifier
    xgb_model = XGBClassifier(eval_metric = "auc", 
                                learning_rate = 0.01, 
                                n_estimators=1000,
                                max_depth=10,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective= 'binary:logistic',
                                nthread=4,
                                scale_pos_weight=80, 
                                device = device)

    # Train the classifier
    xgb_model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = xgb_model.predict(X_train)
    print("Training results")
    cm = confusion_matrix(Y_train.get(), Y_pred)
    print(cm)
    report = classification_report(Y_train.get(), Y_pred, target_names=["no spike", "w spike"], output_dict=True)
    print(report['w spike']['f1-score'])

    Y_pred = xgb_model.predict(X_test)
    print("Testing results")

    return Y_test.get(), Y_pred

def old_classifier_mega(model, train_dataloader, test_dataloader, device):

    test_metric = initialize_test_metrics()
    
    y_pred, y_true = infer.get_raw_error_cards_mega(model, train_dataloader, device)
        
    y_pred = torch.mean(y_pred[:,:,2], dim = 1).squeeze().cpu().numpy()
    y_true = y_true.cpu().numpy()



    bf_performance = bf_search(y_pred, y_true)

    test_metric = set_test_metrics(test_metric, bf_performance)

    TP = bf_performance['TP']
    TN = bf_performance['TN']
    FP = bf_performance['FP']
    FN = bf_performance['FN']

    accuracy = get_accuracy(TP, TN, FP, FN)

    f1_score = bf_performance['F1']

    return accuracy, f1_score

def classifier_mega_features(model, train_dataloader, test_dataloader, device):

    print("----------------------------------------------------")
    print("CLASSIFIER DATASET")

    print(f'Training with')

    X_train, Y_train = infer.get_raw_error_cards_mega(model, train_dataloader, device)
    X_train = X_train[:,:,1:3].squeeze().cpu().numpy()
    X_train = X_train.reshape(-1, 300, 274, 3)
    X_train = X_train[:60:270,:,:]
    X_train = X_train.reshape(-1, 274, 3)
    X_train = (X_train - np.mean(X_train, axis = 0))/np.std(X_train,axis=0)

    Y_train = Y_train.cpu().numpy()
    Y_train = Y_train.reshape(-1, 300)
    Y_train = Y_train[:,60:270].flatten()


    print(f'Testing with')

    X_test, Y_test = infer.get_raw_error_cards_mega(model, test_dataloader, device)
    X_test = X_test[:,:,1:3].squeeze().cpu().numpy()
    Y_test = Y_test.cpu().numpy()


    # X_train_df = pd.DataFrame(X_train)
    # shift = 8
    # X_train_std = X_train_df.rolling(shift).mean().shift(-int(shift/2)).replace(np.nan, 0).rolling(8).max()
    # Y_train_df = pd.DataFrame(Y_train)
    # Y_train = Y_train_df.rolling(12).max().replace(np.nan, 0)

    # X_train_mean = np.std(X_train_std.to_numpy(), axis = 1)
    
    # threshold = 0.50
    # X_train_thres = (X_train_mean > threshold).astype(int)


    # report = classification_report(X_train_thres, Y_train, target_names=["no spike", "w spike"], output_dict=True)
    # f1_score = report['w spike']['f1-score']
    # print("f1score", f1_score)
    # cm = confusion_matrix(X_train_thres, Y_train)
    # print("Confusion matrix", cm)

    # return np.expand_dims(X_train_thres, axis=1), Y_train

    return X_train, Y_train




