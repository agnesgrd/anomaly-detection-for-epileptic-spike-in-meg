# Data Manipulation
import numpy as np
import torch
import csv
import pickle
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# File and System Operations
import os
import os.path as op
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt

# Local Imports
from utils import save_obj, load_obj

def plot_epochs_metric(train, valid, lr, file_path, model_name):
	fig, ax1 = plt.subplots()
	ax1.set_xlabel("epoch", fontsize="large")
	ax1.set_ylabel("loss", fontsize="large")
	ax1.plot(np.linspace(0, len(train), num=len(train)), train, label = "training loss")
	ax1.plot(np.linspace(0, len(valid), num=len(valid)), valid, label = "validation loss")
	# ax1.plot(np.linspace(0, len(test), num=len(test)), test, label = "testing f1 score")
	
	ax1.set_ylim(0, max((max(valid), max(train)))*1.1)
	# ax1.set_yscale('log')

	ax2 = ax1.twinx()
	ax2.plot(np.linspace(0, len(lr), num=len(lr)), lr, label = "learning rate")
	ax1.legend(loc="upper left")
	ax2.legend(loc = "upper right")
	ax2.set_ylim(0,max(lr)*1.1)

	plt.title("model " + model_name)
	plt.savefig(file_path + model_name + ".png", bbox_inches="tight")
	plt.close()

def plot_pca(features, labels, file_path, model_name, simulation_nb):

	features = features.cpu().numpy()
	labels = labels.cpu().numpy()

	chan_num = 36
	sqr = int(chan_num**0.5)

	pca = PCA(n_components=2)
	spikes_ts = (labels == 1)
	labels_spikes = labels[spikes_ts]
	labels_no_spikes = labels[~spikes_ts]

	pca_info = list()
	features_pca_spikes = list()
	features_pca_no_spikes = list()

	for i in tqdm(range(0,273, 254//chan_num), desc = "channel by channel"):
		f = features[:,i,:]
		pca.fit(f)
		features_pca = pca.fit_transform(f)
		pca_info.append(pca)
		features_pca_spikes.append(features_pca[spikes_ts])
		features_pca_no_spikes.append(features_pca[~spikes_ts])

	label_color_dict = {0:'green', 1:'red'}

	fig, ax = plt.subplots(nrows=sqr, ncols=sqr,figsize=(68, 68))

	k = 0
	for i in range(sqr):
		for j in range(sqr):
			ax[i][j].scatter(features_pca_no_spikes[k][:,0], features_pca_no_spikes[k][:,1],
			c=label_color_dict[0], alpha=0.1)
			ax[i][j].scatter(features_pca_spikes[k][:,0], features_pca_spikes[k][:,1],
			c=label_color_dict[1], alpha=0.1)
			ax[i][j].set_title(f'Channel {k*254//chan_num+10}')
			ax[i][j].set_xlabel('PC 1 (%.2f%%)' % (pca_info[k].explained_variance_ratio_[0]*100))
			ax[i][j].set_ylabel('PC 2 (%.2f%%)' % (pca_info[k].explained_variance_ratio_[1]*100)) 
			k+=1


	plt.savefig(file_path + model_name + "_" + simulation_nb + "_PCA.png", bbox_inches="tight")

def plot_lda(features, labels, file_path, model_name, simulation_nb):

	features = features.cpu().numpy()
	labels = labels.cpu().numpy()

	chan_num = 90
	sqr = int(chan_num**0.5)

	lda = LDA(n_components=1)
	spikes_ts = (labels == 1)
	labels_spikes = labels[spikes_ts]
	labels_no_spikes = labels[~spikes_ts]

	pca_info = list()
	features_lda_spikes = list()
	features_lda_no_spikes = list()

	for i in tqdm(range(5,271, 254//chan_num), desc = "channel by channel"):
		print(f'...channel {i}')
		f = features[:,i,:]
		features_lda = lda.fit_transform(f, labels)
		features_lda_spikes.append(features_lda[spikes_ts])
		features_lda_no_spikes.append(features_lda[~spikes_ts])

	label_color_dict = {0:'green', 1:'red'}

	fig, ax = plt.subplots(nrows=sqr, ncols=sqr,figsize=(68, 68))

	k = 0
	for i in range(sqr):
		for j in range(sqr):
			y_no_spikes = np.zeros((features_lda_no_spikes[k].shape[0]))
			y_spikes = np.zeros((features_lda_spikes[k].shape[0]))
			ax[i][j].scatter(features_lda_no_spikes[k][:,0], y_no_spikes, c=label_color_dict[0], alpha=0.1)
			ax[i][j].scatter(features_lda_spikes[k][:,0], y_spikes, c=label_color_dict[1], alpha=0.5)
			ax[i][j].set_title(f'Channel {k*254//chan_num+10}')
			ax[i][j].set_xlabel('LDA 1')
			k+=1


	plt.savefig(file_path + model_name + "_" + simulation_nb + "_LDA.png", bbox_inches="tight")

def plot_heatmap(X, Y, file_path, model_name, simulation_nb):
	win = X.cpu().numpy()
	lab = Y.cpu().numpy()

	np.save('X_train_11.npy', win)
	np.save('Y_train_11.npy', lab)

	mini = 7000
	maxi = 8000
	ticks=np.arange(0,maxi-mini,100)

	for i in range(5):
		plt.figure(figsize = (10, 50))
		plt.subplot(1,2,1)
		plt.title('Anomaly scores')
		plt.imshow(win[mini:maxi,:,i].squeeze(), vmin=0, vmax=1, cmap='seismic', aspect='auto')
		plt.yticks(ticks)
		plt.subplot(1,2,2)
		plt.title('Spikes')
		plt.imshow(np.expand_dims(lab[mini:maxi], axis=1), vmin=0, vmax=1, cmap='seismic', aspect='auto')
		plt.colorbar()
		plt.yticks(ticks)
		plt.savefig(file_path+model_name+'_heatmap_criterion'+str(i)+'.png')

def plot_heatmap_i(X, Y, file_path, model_name, simulation_nb):
	win = X
	lab = Y

	mini = 7000
	maxi = 8000
	ticks=np.arange(0,maxi-mini,100)


	plt.figure(figsize = (10, 50))
	plt.subplot(1,2,1)
	plt.title('Anomaly scores')
	plt.imshow(win[mini:maxi,:], vmin=0, vmax=1, cmap='seismic', aspect='auto')
	plt.yticks(ticks)
	plt.subplot(1,2,2)
	plt.title('Spikes')
	plt.imshow(np.expand_dims(lab[mini:maxi], axis=1), vmin=0, vmax=1, cmap='seismic', aspect='auto')
	plt.colorbar()
	plt.yticks(ticks)
	plt.savefig(f'{file_path} {model_name} _heatmap_criterion{str(2)}_new_84.png')

def plot_contrast(X, Y, file_path, model_name, model_nb, patient):

	X = torch.reshape(X, (-1, 300, 274, 5))
	Y = torch.reshape(Y, (-1, 300))

	X = X[:,60:270,:,:]
	Y = Y[:,60:270]

	X = torch.reshape(X, (-1, 274, 5)).cpu().numpy()
	Y = torch.flatten(Y).cpu().numpy()

	Xs = (X - np.mean(X, axis = 0, keepdims=True))/np.std(X, axis = 0, keepdims = True)

	####### Windows with spike ##############
	Yw = Y.reshape(-1,30)
	spike_mask_w = np.any(Yw==1, axis = 1)

	####### Windows centered on spike ##############
	spike_mask = (Y == 1)
	spike_loc = np.argwhere(Y==1).squeeze()
	ext_spike_loc = spike_loc
	print("Spike number (1 by 1):", len(ext_spike_loc))
	for i in range(1,16):
		ext_spike_loc = np.concatenate((ext_spike_loc, spike_loc - i))
		if i!=15:
			ext_spike_loc = np.concatenate((ext_spike_loc, spike_loc + i))
	print("Spike number (60 by 60):", len(ext_spike_loc))
	ext_spike_loc = np.sort(ext_spike_loc)

	####### X spike ##############
	X_centered_spike = Xs[ext_spike_loc,:,:]
	X_cent_resh_spike = X_centered_spike.reshape((-1, 30, 274, 5))
	X_resh_mean_spike = np.mean(X_cent_resh_spike, axis = 0)

	####### Y spike ##############
	Y_cent_spike = Y[ext_spike_loc]
	Y_cent_resh_spike = Y_cent_spike.reshape((-1, 30))

	####### X no spike ##############
	X_resh = Xs.reshape((-1, 30, 274, 5))
	no_spike_loc = np.argwhere(spike_mask_w == 0).squeeze()
	np.random.shuffle(no_spike_loc)
	no_spike_loc_random = no_spike_loc[0:len(spike_loc)]
	X_resh_mean_no_spike = np.mean(X_resh[no_spike_loc_random,:,:,:], axis = 0)
	
	plt.figure(figsize = (10, 10))

	for i in range(0,5):
		plt.subplot(1,6,i+1)
		plt.title(f'Mean \n score \n spike \n {i}')
		plt.imshow(np.abs(X_resh_mean_spike-X_resh_mean_no_spike)[:,:,i].squeeze(), vmin=0, vmax=2, cmap='seismic', aspect='auto')

		plt.subplot(1,6,6)
		plt.title('Spikes')
		plt.imshow(np.expand_dims(np.mean(Y_cent_resh_spike, axis=0), axis=1), vmin=0, vmax=2, cmap='seismic', aspect='auto')

		plt.colorbar()
		plt.savefig(f'{file_path} spike_vs_nospike_model_{model_nb} _patient_ {patient}_quater.png')





# def save_model_results(file_path, model_name, y_test, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     specificity = tn / (tn + fp)

#     fields = [
#         accuracy_score(y_test, y_pred),
#         f1_score(y_test, y_pred),
#         specificity,
#         recall_score(y_test, y_pred),
#         confusion_matrix(y_test, y_pred).ravel(),
#     ]

#     print(fields)
#     add_header = False
#     if not (os.path.exists(file_path + model_name + "_cvresults.csv")):
#         add_header = True

#     with open(file_path + model_name + "_cvresults.csv", "a", newline="") as f:
#         writer = csv.writer(f)
#         if add_header:
#             writer.writerow(
#                 [
#                     "accuracy",
#                     "f1-score",
#                     "specificity",
#                     "sensitivity",
#                     "confusion matrix",
#                 ]
#             )
#         writer.writerow(fields)


def save_model_predictions(
	X_test_ids, path_extracted_data, file_path, model_name, simulation_nb, y_test, y_pred
):
	win = X_test_ids[:, 0]
	sub = X_test_ids[:, 1]
	lab = X_test_ids[:, 2]

	prevsub = 1000

	for ind, i in enumerate(sub):
		if i != prevsub:
			y_timing_data = load_obj(
				"data_raw_" + str(i).zfill(3) + "_b3_timing.pkl", path_extracted_data[1]
			)
			y_block_data = load_obj(
				"data_raw_" + str(i).zfill(3) + "_b3_blocks.pkl", path_extracted_data[1]
			)
			y_label_data = load_obj(
				"data_raw_" + str(i).zfill(3) + "_b3_labels.pkl", path_extracted_data[1]
			)

		y_timing = y_timing_data[win[ind]]
		y_block = y_block_data[win[ind]]
		y_label = y_label_data[win[ind]]

		add_header = False
		if not (os.path.exists(file_path + model_name + '_' + str(simulation_nb) + "_cvpredictions.csv")):
			add_header = True

		with open(file_path + model_name + '_' + str(simulation_nb) + "_cvpredictions.csv", "a", newline="") as f:
			writer = csv.writer(f)
			if add_header:
				writer.writerow(["subject", "block", "timing", "true", "test", "pred"])
			writer.writerow([i, y_block, y_timing, y_test[ind]==y_pred[ind], y_test[ind], y_pred[ind]])

		prevsub = i


# def save_model_embeddings(
#     reading_embeddings_cnn,
#     reading_labels,
#     reading_outputs,
#     path_writing_data,
#     model_name,
# ):
#     np.save(path_writing_data + model_name + "_labelsh_umap.npy", reading_labels)
#     np.save(
#         path_writing_data + model_name + "_embeddings_cnn_umap.npy",
#         reading_embeddings_cnn,
#     )
#     np.save(
#         path_writing_data + model_name + "_outputsh_umap_balanced_test.npy",
#         reading_outputs,
#     )



