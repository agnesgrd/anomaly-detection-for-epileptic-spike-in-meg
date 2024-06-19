import os
import sys 

################################################# DATA #################################################

# Path to where the binary files containing windows are savec
path_data = "/sps/crnl/aguinard/data/windows_sfreq150_size2"
path_extracted_data = [path_data + "_ae_std/", path_data + "_clf_std/"]

# Path to save the model files and model results
path_writing_data = "/pbs/home/a/aguinard/DeepEpi/anomDetect/results/"

subjects = [
    sub[:-11]
    for sub in os.listdir(path_extracted_data[0])
    if (
        os.path.isfile(os.path.join(path_extracted_data[0], sub))
        and ("b3_windows" in sub)
    )
]
subjects = sorted(subjects)
patient = [50, 83, 97, 18, 84, 11, 105, 42, 20, 30, 101, 117, 4, 65, 100]   # list of patients to use or False for all patients 

sfreq = 150  # sampling frequency of the data in Hz
window_size = 2
dim = (int(sfreq * window_size), 274)  # sample shape
save_emb = False
augmentation = True

################################################# MODEL #################################################
# Name of the model to run, model architectures are defined in the Models folder.
# Non exhaustive list of available models "AE1D", "AE2D" , "WAVE"
# See Model folder to see all possible models
model = sys.argv[2]
batch_size = 200
filters = [32,32,32]

################ WAVELETS ###############
pywt_family = 'db5' #tested with haar, db5
a = 1.0
b = 1.0
c = 1.0
d = 1.0
e = 1.0
t = a + b + c + d + e
a = a/t
b = b/t
c = c/t
d = d/t
e = e/t

################################################# LOSS #################################################
# Non exhaustive list of available loss 'BCEWithLogitsLoss', 'BCELoss'
# Non exhaustive list of available optimizer 'SGD', 'Adam'
criterion = "MSELoss"
optimizer = "Adam"
learning_rate = 0.001  # 0.0001
weight_decay = 0.0005
stepLR = 50 
gammaLR = 0.8


num_epochs = 100
n_epochs_stop = 10

# Warm up / step pour la modification du learning rate
warmup_epochs = 10
beta = 0.1
step_size = 3

################################################# CLASSIFIER #################################################
# Non exhaustive list of available classifier 'xgboost', "svm", "NN"
classifier = "mega"
model_nb = 84 #retrieve best model for testing


