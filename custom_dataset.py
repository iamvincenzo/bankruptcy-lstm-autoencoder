import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

""" Function used to import data from csv files. """
def get_data(data_path, seq_len, train_only_ae, verbose=False):
    print(f"\nLoading data...")

    # import data into dataframe
    df_train = pd.read_csv(os.path.join(data_path, "training_ready.csv"))
    df_valid = pd.read_csv(os.path.join(data_path, "validation_ready.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test_ready.csv"))

    if train_only_ae:
        # removes 'failed' company from the data
        df_train = df_train[df_train.status_label == 'alive']
        df_valid = df_valid[df_valid.status_label == 'alive']
        df_test = df_test[df_test.status_label == 'alive']
        
    # remove useless columns
    df_train.drop(columns=["fyear", "cik", "status_label"], inplace=True)
    df_valid.drop(columns=["fyear", "cik", "status_label"], inplace=True)
    df_test.drop(columns=["fyear", "cik", "status_label"], inplace=True)

    if verbose:
        print(f"\ndf_train: \n{df_train.head(5)}\n")
        print(f"\ndf_valid: \n{df_valid.head(5)}")
        print(f"\ndf_test: \n{df_test.head(5)}")

        print(f"\ndf_train-shape: \n{df_train.shape}")
        print(f"\ndf_valid-shape: \n{df_valid.shape}")
        print(f"\ndf_test-shape: \n{df_test.shape}")

        print(f"\ntotal train-samples: \n{df_train.shape[0]//seq_len}")
        print(f"\ntotal valid-samples: \n{df_valid.shape[0]//seq_len}")
        print(f"\ntotal test-samples: \n{df_test.shape[0]//seq_len}")

    # convert data into numpy array
    np_train = df_train.to_numpy(copy=False, dtype=np.float32)
    np_valid = df_valid.to_numpy(copy=False, dtype=np.float32)
    np_test = df_test.to_numpy(copy=False, dtype=np.float32)

    if verbose:
        print(f"\nnp_train-shape: \n{np_train.shape}")
        print(f"\nnp_valid-shape: \n{np_valid.shape}")
        print(f"\nnp_test-shape: \n{np_test.shape}")

    return np_train, np_valid, np_test


""" Custom class used to create the training, validation and test sets. """
class CustomDataset(Dataset):
    def __init__(self, x, seq_len):
        super().__init__()
        self.x = x
        self.seq_len = seq_len

    def __getitem__(self, index):
        # index for rows selection
        start_index = index * self.seq_len
        end_index = start_index + self.seq_len

        # extract the first 18 columms
        data_x = self.x[start_index:end_index, :-1]
        # extract the last (19) column
        data_y = self.x[start_index, -1:]

        tensor_x = torch.from_numpy(data_x)
        tensor_y = torch.from_numpy(np.array([data_y, 1-data_y]))

        return tensor_x, tensor_y

    def __len__(self):
        return len(self.x) // self.seq_len
    