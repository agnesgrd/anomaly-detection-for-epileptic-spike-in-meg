# DETECTION OF INTERICTAL EPILEPTIFORM DISCHARGES IN MAGNETOENCEPHALOGRAPHIC RECORDINGS VIA DEEP NEURAL NETWORKS
## Raw data
On the ccin2p3 here: /sps/crnl/pmouches/data/MEG_PHRC_2006_Dec23

Raw data are formatted as follows:

data_raw_N.pkl: Contains the raw data as a dictionary. Each dictionary entry is a list of lists. Patient recordings are split into ~15 files of 3minutes each. Each list corresponds to one recording file (which name is stored in data['file']). 
- data['meg']: raw meg signal values
- data['events']: contain timing of spike events (original annotations)
- data['reannot']: contain timing of spike events after reannotations (if reannotation was performed, empty otherwise)
- data['file']: contains the file name (.ds = ctf file format) corresponding to the data

## Windowing data before model training:
*create_windows_ae.py:* Script to generate subfiles (cropped windows) data_raw_N_b3_windows_bi, data_raw_N_b3_labels, data_raw_N_b3_blocks, data_raw_N_b3_timing, data_raw_N_b3_labels_t used to train/validate the autoencoder (NORMAL SIGNAL ONLY).
*create_windows_clf.py:* Script to generate subfiles (cropped windows) data_raw_N_b3_windows_bi, data_raw_N_b3_labels, data_raw_N_b3_blocks, data_raw_N_b3_timing, data_raw_N_b3_labels_t used to train/test the classifer (NORMAL SIGNAL + SPIKES).
This script crops windows according to a set of window shape parameters, check if the window contains a spike, and saves the window data, label, timing (center of the window), block number (= # of the recording file) and label by timestep (rectangular-shaped pulse around the spikes). 
Options in the script allows to:
-	Apply center-scaling on each window before saving
-	Discard windows with artefacts (high signal deviation from the mean) /!\ this results in discarding some spikes with high amplitude. Might be better to remove it?
-	Include only blocks (= recording files) with the most annotated spikes to (1) speed up training and (2) reduce imbalance ratio. I am currently using 3 best blocks per participant, which seems to give good results and the distribution shift between training and testing data does not seem to harm the model performance at test time.
-	/!\ some participants have no spike at all. Until now, I’ve discarded these patients (never used them for training or testing). There data_raw_N.pkl files are on the cluster, it’s up to you to add a few lines of code to avoid these patients if you don’t want to include them neither.

## anomDetect folder:
Need *requirements_tfvenv.txt*.
*autoencoder_model.py:* Contains functions needed for:
-	Data generation 
-	Custom metrics implementation
-	Model architecture definition
-	Model training
-	Model testing
*model_training.py:* Main script

## anomDetectTORCH folder:
- Models folder: contains implemented models
    - AE1D
    - AE2D
    - AE1D with a smaller latent space
    - AE2D with a smaller latent space
    - MEGA (wavelets)
- *main_train.py:* main script, allows us to train and validate the autoencoder, then train and test the classifier
- *data_processing.py* and *TimeSeries DataSet.py:* contains the data load class specific to our implemented Pytorch models
- *loop.py:* contains the training and validation loop for the autoencoder
- *classification.py:* contains the training and testing functions for the classifier
- *saving_functions.py:* contains function for saving and plotting the results (saving as csv files and plotting with matplotlib)
- *utils.py:* contains some utils function
- *params.py:* contains all the modifiable hyperparameters
