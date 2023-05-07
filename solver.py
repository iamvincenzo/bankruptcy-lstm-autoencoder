import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from model import LSTMAutoencoder

""" Class solver for training and testing. """
class Solver(object):
    """ Initialize configuration. """
    def __init__(self, device, num_epochs, train_loader, valid_loader):
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.seq_len = 5
        self.n_features = 18
        self.embedding_dim = 128
        self.input_dim = 128

        self.model = LSTMAutoencoder(seq_len=self.seq_len, n_features=self.n_features, 
                                     embedding_dim=self.embedding_dim, input_dim=self.input_dim)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.criterion = nn.L1Loss(reduction='sum').to(device)

        self.history = dict(train=[], val=[])
    
    """ Mehtod used to train the model. """
    def train_model(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())

        best_loss = np.Inf

        self.model.train()
        
        for epoch in range(self.num_epochs):           
            train_losses = []

            train_loop = tqdm(self.train_loader, total=len(self.train_loader), leave=True)
            
            for _, seq_true in train_loop:
                self.optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                loss = self.criterion(seq_pred, seq_true)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = self.validate_model()

            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print(f"Epoch [{epoch + 1}/{self.num_epochs + 1}]: "
                  f"train loss {train_loss} val loss {val_loss}")
        
        self.model.load_state_dict(best_model_wts)
        
        return self.model.eval(), self.history

    """ Method used to validate the model. """
    def validate_model(self):
        val_losses = []            
        
        self.model.eval()

        val_loop = tqdm(self.valid_loader, total=len(self.valid_loader), leave=True)
        
        with torch.no_grad():
            for _, seq_true in val_loop:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                loss = self.criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        self.model.train()

        return val_loss
    