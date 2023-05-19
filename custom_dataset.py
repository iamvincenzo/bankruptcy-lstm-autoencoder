import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, x, seq_len):
        super().__init__()
        self.x = x
        self.seq_len = seq_len

    def __getitem__(self, index):
        start_index = index * self.seq_len
        end_index = start_index + self.seq_len
        data_x = self.x[start_index:end_index, :]
        tensor_x = torch.from_numpy(data_x)

        return tensor_x

    def __len__(self):
        return len(self.x) // self.seq_len