import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __getitem__(self, index):
        pass

    def get_lenght(self):
        pass