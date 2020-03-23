from six.moves import cPickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import json

with open("util/korean_labels.json", 'r', encoding='UTF-8-sig') as f:
    char2label = json.load(f)


def load_dataset(data_path, **kwargs):
    with open(data_path, 'rb') as cPickle_file:
        X_train, y_train, X_valid, y_valid, X_test, y_test = cPickle.load(cPickle_file)
    for data in [X_train, y_train, X_valid, y_valid, X_test, y_test]:
        assert len(data) > 0
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Input x: list of np array with shape (timestep, feature)
# Return new_x : a np array of shape (len(x), padded_timestep, feature)
def zero_padding(x, pad_len):
    features = x[0].shape[-1]
    new_x = np.zeros((len(x), pad_len, features))
    for idx, ins in enumerate(x):
        new_x[idx,:len(ins),:] = ins
    return new_x

# A transfer function for LAS label
# We need to collapse repeated label and make them onehot encoded
# each sequence should end with an <eos> (index = 1)
# Input y: list of np array with shape ()
# Output tuple: (indices, values, shape)
def one_hot_encode(Y, max_len, max_idx):
    new_y = np.zeros((len(Y), max_len, max_idx+2))
    for idx, label_seq in enumerate(Y):
        for cnt, label in enumerate(label_seq):
            new_y[idx, cnt, label+2] = 1.0
        new_y[idx, cnt+1, 1] = 1.0 # <eos>
    return new_y

class KsponDataset(Dataset):
    def __init__(self, X, Y, max_timestep, max_label_len, bucketing):
        self.X = zero_padding(X, max_timestep)
        self.Y = one_hot_encode(Y, max_label_len, len(char2label))

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return len(self.X)


def create_dataloader(X, Y, max_timestep, max_label_len, batch_size, shuffle, bucketing, **kwargs):
    return DataLoader(KsponDataset(X, Y, max_timestep, max_label_len, bucketing),
                      batch_size=batch_size, shuffle=shuffle)
