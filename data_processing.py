import numpy as np
from sklearn.model_selection import train_test_split


def generate_database(labels, sub, set, rs):

    ############# This functions filling the "ids" array which contains a line for each window with: ids[N,0]=id of the window (in subject space), ids[N,1]=id of the subject, ids[N,2]=label

    # Go through all subjects and add their number of windows to "total_nb_window"
    nb_windows = len(labels)
    ids = np.zeros((nb_windows, 3), dtype=int)
    ids[:,0] = np.arange(0, nb_windows, 1)
    ids[:,1] = int(sub)
    ids[:,2] = np.array(labels)

    # From the "ids" array, extract valid, test and train subjects based on data_valid,data_test,data_train
    if set == 'ae':
        ids = ids[np.argwhere(ids[:, 2] == 0)].squeeze()

    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=rs)

    return train_ids, test_ids

