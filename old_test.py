from utils import load_obj
import params
from TimeSeriesDataSet import TimeSeriesDataSet
from data_processing import generate_database
from torch.utils.data import DataLoader
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import pywt
import matplotlib.pyplot as plt

subjects = sorted(params.subjects)


Y = list()
data_all = list()
for p in params.patient:
    data_train = load_obj(
        "data_raw_" + str("{:03d}".format(p)) + "_b3_labels.pkl",
        params.path_extracted_data[0],
    )
    data_test = load_obj(
        "data_raw_" + str("{:03d}".format(p)) + "_b3_labels.pkl",
        params.path_extracted_data[1],
    )
    Y.append([data_train, data_test])
    data_all.append(p)

data_train = data_all
data_valid = data_all
data_test = data_all

print("data_all: ", data_all)
print("data_train: ", data_train)
print("data_valid: ", data_valid)
print("data_test: ", data_test)
print("labels: ", len(Y))

print("window labels loaded")

# Prepare dataset et create data iterator
X_train_ids, X_test_ids, X_valid_ids = generate_database(
    Y, data_all, data_valid, data_test, data_train, rs=0
)

train_window_dataset = TimeSeriesDataSet(
    X_train_ids.tolist(),
    params.dim,
    params.path_extracted_data[0],
    out="XX",
    aug=params.augmentation,
)
train_dataloader = DataLoader(
    train_window_dataset, batch_size=params.batch_size, shuffle=True
)

valid_window_dataset = TimeSeriesDataSet(
    X_valid_ids.tolist(), params.dim, params.path_extracted_data[1], out="Xy"
)
valid_dataloader = DataLoader(
    valid_window_dataset, batch_size=params.batch_size, shuffle=True
)

test_window_dataset = TimeSeriesDataSet(
    X_test_ids.tolist(), params.dim, params.path_extracted_data[1], out="Xy"
)
test_dataloader = DataLoader(
    test_window_dataset, batch_size=params.batch_size, shuffle=False
)

X_train, _ = next(iter(train_dataloader))
X_test, Y_test = next(iter(test_dataloader))

X = X_train[40]
xfm = DWT1DForward(J=1, wave='haar', mode='zero')
Yl, Yh = xfm(X)

print(Yl.shape)
print(Yh[0].shape)
print(len(Yh))
#print(Yh[1].shape)
#print(Yh[2].shape)

cA, cD = pywt.dwt(X, wavelet='haar', mode='constant')

plt.figure()
plt.subplot(121)
plt.imshow(Yl.squeeze())
plt.title('Input Signal')
plt.subplot(122)
plt.imshow(cA.squeeze())
plt.title('Wavelet Decomposition')
plt.savefig('/pbs/home/a/aguinard/DeepEpi/anomDetect/results/DWTt.png')

