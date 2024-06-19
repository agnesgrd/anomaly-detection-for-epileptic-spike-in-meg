# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

# Sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data Manipulation
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# Visualization

# File and System Operations
import os
import sys

# Utilities
import gc
import random

# Local Imports
import classification as clf
import inference as infer
from Models import AE1D, AE2D, AE2D_smaller, AE1D_smaller, multiWaveGCUNet
import saving_functions as save
import params
from TimeSeriesDataSet import TimeSeriesDataSet
import utils
from data_processing import generate_database

######################### CLASSIFYING ##########################

def train_classifier_classic(model, train_dataloader, test_dataloader, device, simulation_nb):
    
    # Classifier using error cards
    # if params.classifier == "xgboost":
    #     spike_gt, spike_pred = clf.classifier_xgboost(model, train_dataloader, test_dataloader, device)
    # elif params.classifier == "svm":
    #     spike_gt, spike_pred= clf.classifier_svm(model, train_dataloader, test_dataloader, device)
    # elif params.classifier == "nn":
    #     spike_gt, spike_pred = clf.classifier_nn(model, train_dataloader, test_dataloader, device, simulation_nb)
        
    spike_gt, spike_pred = clf.classifier_mega_nn(model, train_dataloader, test_dataloader, simulation_nb, device)

    accuracy = accuracy_score(spike_gt, spike_pred)
    cm = confusion_matrix(spike_gt, spike_pred)
    print("Confusion matrix", cm)
    report = classification_report(spike_gt, spike_pred, target_names=["no spike", "w spike"], output_dict=True)
    f1_score = report['w spike']['f1-score']

    print("----------------------------------------------------")
    print("Testing accuracy:", accuracy, "\ f1-score (w spike):", f1_score)

def train_classifier_mega(model, train_dataloader, test_dataloader, device, simulation_nb):

    # More complex classifier
    if params.classifier == "mega":
        accuracy, f1_score = clf.old_classifier_mega(model, train_dataloader, test_dataloader, device)

    print("----------------------------------------------------")
    print("Testing accuracy:", accuracy, "\ f1-score (w spike):", f1_score)


def train_classifier_features(model, train_dataloader, test_dataloader, device, simulation_nb):
    
    X_train, Y_train = clf.classifier_mega_features(model, train_clf_dataloader, test_clf_dataloader, device)
    save.plot_heatmap_i(X_train, Y_train, params.path_writing_data, "MEGA", simulation_nb)
    print("PCA saved")

def train_classifier_latent(model, train_dataloader, test_dataloader, device, simulation_nb):

    if params.classifier == "mega":
        spike_gt, spike_pred = clf.classifier_mega_xgboost_latent(model, train_clf_dataloader, test_clf_dataloader, device)

    accuracy = accuracy_score(spike_gt, spike_pred)
    cm = confusion_matrix(spike_gt, spike_pred)
    print("Confusion matrix", cm)
    report = classification_report(spike_gt, spike_pred, target_names=["no spike", "w spike"], output_dict=True)
    f1_score = report['w spike']['f1-score']

    print("----------------------------------------------------")
    print("Testing accuracy:", accuracy, "\ f1-score (w spike):", f1_score)

######################### VISUALIZATION ##########################
def display_pca(model, test_dataloader, device, simulation_nb ):
    print("Starting computing PCA")
    if params.classifier == "mega":
        (latent1, latent2, latent3), y_true = infer.get_latent_features_mega(model, test_dataloader, device)
        print("Starting to compute PCA of latent space 1")
        save.plot_pca(latent1, y_true, params.path_writing_data, "MEGA_latent1", simulation_nb)
        print("Starting to compute PCA of latent space 2")
        save.plot_pca(latent2, y_true, params.path_writing_data, "MEGA_latent2", simulation_nb)
        print("Starting to compute PCA of latent space 3")
        save.plot_pca(latent3, y_true, params.path_writing_data, "MEGA_latent3", simulation_nb)

def display_lda(model, test_dataloader, device, simulation_nb):
    print("Starting computing LDA")
    if params.classifier == "mega":
        (latent1, latent2, latent3), y_true = infer.get_latent_features_mega(model, test_dataloader, device)
        print("Starting to compute LDA of latent space 1")
        save.plot_lda(latent1, y_true, params.path_writing_data, "MEGA_latent1", simulation_nb)
        print("Starting to compute LDA of latent space 2")
        save.plot_lda(latent2, y_true, params.path_writing_data, "MEGA_latent2", simulation_nb)
        print("Starting to compute LDA of latent space 3")
        save.plot_lda(latent3, y_true, params.path_writing_data, "MEGA_latent3", simulation_nb)

def display_heatmap(model, test_dataloader, simulation_nb):
    print("Starting to compute anomaly scores")
    X_test, Y_test = infer.get_raw_error_cards_mega(model, test_dataloader, device)
    save.plot_heatmap(X_test, Y_test, params.path_writing_data, "MEGA", simulation_nb)
    print("PCA saved")

def display_contrast(model, dataloader, simulation_nb, p):
    print("Starting to compute contraste")
    X, Y = infer.get_raw_error_cards_mega(model, dataloader, device)
    save.plot_contrast(X, Y, params.path_writing_data, "MEGA", params.model_nb, p)











######################### MAIN TEST ##########################

torch.manual_seed(43)


subjects = sorted(params.subjects)

simulation_nb= str(params.model_nb) + '_' + str(np.random.randint(1000))

rs = 42

p = params.patient
ignore = [1, 3, 7, 10, 12, 13, 14, 16, 17, 21, 43, 48, 49, 50, 55, 59, 60, 63, 70, 80, 82, 88, 89, 90, 91, 98, 102, 110, 114, 120, 121]
prange = [i for i in range(len(subjects)) if i not in ignore]
prange = [4]
for p in prange:
    if type(p) == int:
        labels_ae = utils.load_obj(
            "data_raw_" + str("{:03d}".format(p)) + "_b3_labels.pkl",
            params.path_extracted_data[0],
        )
        labels_clf = utils.load_obj(
            "data_raw_" + str("{:03d}".format(p)) + "_b3_labels.pkl",
            params.path_extracted_data[1]
        )

        # Prepare dataset et create data iterator
        train_ae_ids, valid_ae_ids = generate_database(labels_ae, p, "ae", rs)
        train_clf_ids, test_clf_ids = generate_database(labels_clf, p, "clf", rs)

    if type(p) == list:
        train_ae_ids = np.array([], dtype=int).reshape(0,3)
        valid_ae_ids = np.array([], dtype=int).reshape(0,3)
        train_clf_ids = np.array([], dtype=int).reshape(0,3)
        test_clf_ids = np.array([], dtype=int).reshape(0,3)
        for ind, sub in enumerate(sorted(p)):
            labels_ae = utils.load_obj(
            "data_raw_" + str("{:03d}".format(sub)) + "_b3_labels.pkl",
            params.path_extracted_data[0],
            )
            labels_clf = utils.load_obj(
                "data_raw_" + str("{:03d}".format(sub)) + "_b3_labels.pkl",
                params.path_extracted_data[1]
            )
            
            # Prepare dataset et create data iterator

            train_ae_ids_p, valid_ae_ids_p = generate_database(labels_ae, sub, "ae", rs)
            train_clf_ids_p, test_clf_ids_p = generate_database(labels_clf, sub, "clf", rs)
            train_ae_ids = np.vstack((train_ae_ids, train_ae_ids_p))
            valid_ae_ids = np.vstack((valid_ae_ids, valid_ae_ids_p))
            train_clf_ids = np.vstack((train_clf_ids, train_clf_ids_p))
            test_clf_ids = np.vstack((test_clf_ids, test_clf_ids_p))

    print("----------------------------------------------------")
    print("Database generated")
    print("Studied patient(s):", p)
    print("----------------------------------------------------")
    print("AUTOENCODER DATASET")
    print(f"Training with {train_ae_ids.shape[0]} windows")
    print(f"Validation with {valid_ae_ids.shape[0]} windows")
    print(f"(windows of shape {params.window_size} s = {params.sfreq*params.window_size} timestep x 274 channels)")
    print("----------------------------------------------------")

    train_clf_window_dataset = TimeSeriesDataSet(
        train_clf_ids.tolist(), params.dim, params.path_extracted_data[1], out="Xy"
    )

    train_clf_dataloader = DataLoader(
        train_clf_window_dataset, batch_size=params.batch_size, shuffle=True
    )

    test_clf_window_dataset = TimeSeriesDataSet(
        test_clf_ids.tolist(), params.dim, params.path_extracted_data[1], out="Xy"
    )
    test_clf_dataloader = DataLoader(
        test_clf_window_dataset, batch_size=params.batch_size, shuffle=True
    )

    print("Windows loaded")

    # GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Instantiate model
    if params.model == "wavelets":
        model = multiWaveGCUNet.MultiWaveGCUNet(input_channel=274, embedding_dim=64, top_k=20,
                    input_node_dim=2, graph_alpha=3, device=device, gc_depth=1, batch_size=params.batch_size, filters = params.filters)
        model_name = "wavelets"

    if params.criterion == "MSELoss":
        criterion = nn.MSELoss()

    model.to(device)
    criterion.to(device)

    if params.model != "wavelets":
        print(summary(model, (1, 274, 300)))

    # Load the parameters into the model
    utils.resume(model, params.path_writing_data + "best_" + model_name + "_" + str(params.model_nb) + ".pth", device)

    print("Model weights loaded")

    if sys.argv[1] == "classic":

        train_classifier_classic(model, train_clf_dataloader, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "mega":

        train_classifier_mega(model, train_clf_dataloader, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "classiflat":

        train_classifier_latent(model, train_clf_dataloader, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "test":

        train_classifier_features(model, train_clf_dataloader, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "pca":

        display_pca(model, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "lda":

        display_lda(model, test_clf_dataloader, device, simulation_nb)

    elif sys.argv[1] == "heatmap":

        display_heatmap(model, train_clf_dataloader, simulation_nb)

    elif sys.argv[1] == "contrast":

        display_contrast(model, train_clf_dataloader, simulation_nb, p)

