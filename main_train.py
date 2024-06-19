# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

# Sklearn
from sklearn.metrics import accuracy_score, classification_report

# Data Manipulation
import numpy as np

# Visualization

# File and System Operations
import os
import sys

# Utilities
import gc
import random

# Local Imports
from loop import train_epoch, validate_epoch, train_mega_epoch, validate_mega_epoch
from Models import AE1D, AE2D, AE2D_smaller, AE1D_smaller, multiWaveGCUNet
import saving_functions as save
import params
from TimeSeriesDataSet import TimeSeriesDataSet
import utils
from data_processing import generate_database

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# import gc
# gc.collect()
# torch.cuda.empty_cache()

def model_train_test(
    X_test_clf_ids, train_ae_dataloader, valid_ae_dataloader, train_clf_dataloader, test_clf_dataloader, simulation_nb, ft = False
):
    
    # GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Instantiate model
    if params.model == "AE1D":
        model = AE1D.Autoencoder("time", params.sfreq, params.window_size, 0)
        model_name = "AE1D"
    if params.model == "AE1D_smaller":
        model = AE1D_smaller.Autoencoder("time", params.sfreq, params.window_size, 0)
        model_name = "AE1D_smaller"
    elif params.model == "AE2D":
        model = AE2D.Autoencoder("time", params.sfreq, params.window_size)
        model_name = "AE2D"
    elif params.model == "AE2D_smaller":
        model = AE2D_smaller.Autoencoder("time", params.sfreq, params.window_size)
        model_name = "AE2D_smaller"
    elif params.model == "wavelets":
        model = multiWaveGCUNet.MultiWaveGCUNet(input_channel=274, embedding_dim=64, top_k=20,
                   input_node_dim=2, graph_alpha=3, device=device, gc_depth=1, batch_size=params.batch_size, filters = params.filters)
        model_name = "wavelets"
    
    

    
    # Instantiate training parameters
    if params.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.stepLR, gamma=params.gammaLR)

    if params.criterion == "MSELoss":
        criterion = nn.MSELoss()

    model.to(device)
    criterion.to(device)

    if params.model != "wavelets":
        print(summary(model, (1, 274, 300)))

    # Training loop
    loss_valid_all = []
    loss_train_all = []
    f1_score_all = []
    lr_all = []

    if ft:
        model_path = params.path_writing_data +"best_" + model_name + "_" + str(params.model_nb) + ".pth"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_valid_all = checkpoint['validation loss']
        loss_train_all = checkpoint['training loss']
        lr_all = checkpoint['learning rate']

    print("Starting training")

    for epoch in range(params.num_epochs):
        print("Epoch:", epoch+1, "/", params.num_epochs)

        #########################Training##########################
        if params.model =="wavelets":
            training_loss = train_mega_epoch(
                model, train_ae_dataloader, optimizer, scheduler, criterion, device
            )

        else:
            training_loss = train_epoch(
                model, train_ae_dataloader, optimizer, criterion, device
            )

        print("Training loss:", training_loss)

        #########################Validation##########################       

        best_val_loss = 100000

        if params.model =="wavelets":
            validation_loss = validate_mega_epoch(model, valid_ae_dataloader, criterion, device)
        else:
            validation_loss = validate_epoch(model, valid_ae_dataloader, criterion, device)

        print("Validation loss:", validation_loss)

        #########################Classification##########################

        # if epoch%10==0:
        #     if params.classifier == "xgboost":
        #         spike_gt, spike_pred = classifier_xgboost(model, train_clf_dataloader, test_clf_dataloader, device)
        #     elif params.classifier == "svm":
        #         spike_gt, spike_pred= classifier_svm(model, train_clf_dataloader, test_clf_dataloader, device)
        #     elif params.classifier == "nn":
        #         spike_gt, spike_pred = classifier_nn(model, train_clf_dataloader, test_clf_dataloader, simulation_nb, device)
        #     elif params.classifier == "mega":
        #         accuracy, f1_score = old_classifier_mega(model, train_clf_dataloader, test_clf_dataloader, criterion, simulation_nb, device)
            
        # if params.classifier != "mega":
        #     accuracy = accuracy_score(spike_gt, spike_pred)
        #     report = classification_report(spike_gt, spike_pred, target_names=["no spike", "w spike"], output_dict=True)
        #     f1_score = report['w spike']['f1-score']
        
        # print("Testing accuracy:", accuracy, "\ f1-score (w spike):", f1_score)

            #########################Metrics##########################
        # f1_score = 0
        loss_train_all.append(training_loss)
        loss_valid_all.append(validation_loss)
        # f1_score_all.append(f1_score)
        lr_all.append(scheduler.get_lr()[0])

        if validation_loss < best_val_loss:
            utils.checkpoint(model, epoch, optimizer, loss_train_all, loss_valid_all, lr_all, params.path_writing_data + "best_" + model_name + "_" + str(simulation_nb) + ".pth")
            best_val_loss = validation_loss

        save.plot_epochs_metric(
        loss_train_all, loss_valid_all, lr_all, params.path_writing_data, model_name + "_" + str(simulation_nb)
    )
        
  
    print("Finished Training")

    utils.resume(model, params.path_writing_data + "best_" + model_name + "_" + str(simulation_nb) + ".pth", device)

    print("Wait until results are saved")

    save.plot_epochs_metric(
        loss_train_all, loss_valid_all, f1_score_all, lr_all, params.path_writing_data, model_name + "_" + str(simulation_nb)
    )





# save.save_model_predictions(X_test_clf_ids, params.path_extracted_data, params.path_writing_data, model_name, simulation_nb, spike_gt,spike_pred)

#   save.save_model_results(path_writing_data_repeated,model_name,true_spike_list, detected_spike_list_thresh,fold)
    
#   if params.save_emb:
#     save.save_model_embeddings(reading_embeddings_cnn, reading_labels, reading_outputs,path_writing_data_repeated,model_name,fold)

#   true_spike_list, detected_spike_list_raw, detected_spike_list_thresh, reading_embeddings_cnn, reading_labels, reading_outputs = test_model(model, test_dataloader_balanced, device, need_features=params.need_features)

#   save.save_model_results(path_writing_data_repeated,model_name+"_balanced_test",true_spike_list, detected_spike_list_thresh,fold)
#   save.save_model_predictions(X_test_ids_balanced,params.path_extracted_data,path_writing_data_repeated,model_name+"_balanced_test",true_spike_list,detected_spike_list_raw)
#   if params.save_emb:
#       save.save_model_embeddings(reading_embeddings_cnn, reading_labels, reading_outputs,path_writing_data_repeated,model_name+"_balanced_test",fold)


torch.manual_seed(43)

#########################Get subject list##########################
subjects = sorted(params.subjects)

simulation_nb= np.random.randint(1000)

rs = 42

# if params.patient is False:
#     Y = list()
#     data_all = list()

#     for ind, sub in enumerate(sorted(subjects)):
#         data_train = load_obj(sub + "_labels.pkl", params.path_extracted_data[0])
#         data_test = load_obj(sub + "_labels.pkl", params.path_extracted_data[1])
#         Y.append([data_train, data_test])
#         data_all.append(sub.split("_")[2])

#     data_all = list(map(int, data_all))
#     shuffled_data = random.shuffle(data_all)
#     data_train = shuffled_data[: int(0.8 * len(shuffled_data))]
#     data_valid = shuffled_data[
#         int(0.8 * len(shuffled_data)) : int(0.9 * len(shuffled_data))
#     ]
#     data_test = shuffled_data[int(0.9 * len(shuffled_data)) :]

p = params.patient
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
    

print("Database generated:")
print("Studied patient(s):", p)
print("Autoencoder training with", train_ae_ids.shape[0], "windows of shape", str(params.sfreq*params.window_size), "x number of channels 274")
print("Autoencoder validation with", valid_ae_ids.shape[0], "windows of shape", str(params.sfreq*params.window_size), "x number of channels 274")
print("Classifier training with", train_clf_ids.shape[0], "windows of shape", str(params.sfreq*params.window_size), "x number of channels 274")
print("Classifier testing with", test_clf_ids.shape[0], "windows of shape", str(params.sfreq*params.window_size), "x number of channels 274")

train_ae_window_dataset = TimeSeriesDataSet(
    train_ae_ids.tolist(),
    params.dim,
    params.path_extracted_data[0],
    out="XX",
    aug=params.augmentation,
)
train_ae_dataloader = DataLoader(
    train_ae_window_dataset, batch_size=params.batch_size, shuffle=True
)

valid_ae_window_dataset = TimeSeriesDataSet(
    valid_ae_ids.tolist(), params.dim, params.path_extracted_data[0], out="XX"
)
valid_ae_dataloader = DataLoader(
    valid_ae_window_dataset, batch_size=params.batch_size, shuffle=True
)

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

if sys.argv[1] == "auto":

    model_train_test(test_clf_ids, train_ae_dataloader, valid_ae_dataloader, train_clf_dataloader, test_clf_dataloader, simulation_nb)

if sys.argv[1] == "ft":

    model_train_test(test_clf_ids, train_ae_dataloader, valid_ae_dataloader, train_clf_dataloader, test_clf_dataloader, simulation_nb, ft=True)