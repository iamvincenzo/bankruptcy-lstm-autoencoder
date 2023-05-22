import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        print(f"\ndf_train-shape: \n{df_train.shape}")
        print(f"\ndf_valid-shape: \n{df_valid.shape}")
        print(f"\ndf_test-shape: \n{df_test.shape}")

    # convert data into numpy array
    np_train = df_train.to_numpy(copy=False, dtype=np.float32)
    np_valid = df_valid.to_numpy(copy=False, dtype=np.float32)
    np_test = df_test.to_numpy(copy=False, dtype=np.float32)

    if verbose:
        print(f"\nnp_train-shape: \n{np_train.shape}")
        print(f"\nnp_valid-shape: \n{np_valid.shape}")
        print(f"\nnp_test-shape: \n{np_test.shape}")

    print(f"\ntotal train-samples: \n{np_train.shape[0]//seq_len}")
    print(f"\ntotal valid-samples: \n{np_valid.shape[0]//seq_len}")
    print(f"\ntotal test-samples: \n{np_test.shape[0]//seq_len}")

    print(f"\nLoading data Done...")

    return np_train, np_valid, np_test

""" Function used to obtain a dataloader containing 
    only failed companies of the validation set."""
def get_valid_failed_dataloader(data_path, seq_len, batch_size, workers):
    print(f"\nLoading validation data with only failed companies...")
    # import data into dataframe
    df_valid = pd.read_csv(os.path.join(data_path, "validation_ready.csv"))
    # removes 'alive' company from the data
    df_valid = df_valid[df_valid.status_label == 'failed']        
    # remove useless columns
    df_valid.drop(columns=["fyear", "cik", "status_label"], inplace=True)
    # convert data into numpy array
    np_valid = df_valid.to_numpy(copy=False, dtype=np.float32)
    print(f"\ntotal valid-samples: \n{np_valid.shape[0]//seq_len}")
    
    # dataset creation
    valid_failed_dataset = CustomDataset(x=np_valid, seq_len=seq_len)

    # dataloader creation
    valid_failed_loader = DataLoader(dataset=valid_failed_dataset, batch_size=batch_size, 
                                     num_workers=workers, shuffle=False)
    
    print(f"\nLoading data with only failed companies Done...")
    
    return valid_failed_loader


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
        tensor_y = torch.from_numpy(data_y)

        return tensor_x, tensor_y

    def __len__(self):
        return len(self.x) // self.seq_len
    