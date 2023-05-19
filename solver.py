import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from metrics import custom_loss
from pytorchtools import EarlyStopping
from pltotting_utils import plot_losses


""" Class for training, validation and testing. """
class Solver(object):
    """ Initialize configuration. """
    def __init__(self, model, train_loader, valid_loader, test_loader, writer, device, args):
        super().__init__()
        self.args = args
        self.model_name = f"lstm_encdec_{self.args.model_name}.pt"

        self.model = model.to(device)

        # load a pretrained model
        if self.args.resume_train:
            self.model = self.load_model(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), 
                                    lr=self.args.lr, betas=(0.9, 0.999))

        self.num_epochs = args.num_epochs
        self.patience = self.args.patience
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.device = device

        # visualize the model we built on tensorboard
        X = next(iter(self.train_loader))
        self.writer.add_graph(self.model, X.to(self.device))
        self.writer.close()       

    """ Method used to load a model. """
    def load_model(self, device):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.model.load_state_dict(torch.load(check_path, 
                                              map_location=torch.device(device=device)))
        print("\nModel loaded...")

    """ Method used to train the model with early stopping implementation. """
    def train_model(self):
        print(f"\nStarting training...")

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # initialize the early_stopping object
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        early_stopping = EarlyStopping(patience=self.patience, 
                                       verbose=True, path=check_path)

        # put the model in training mode
        self.model.train()

        # loop over the dataset multiple times
        for epoch in range(self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]\n")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch, X in loop:
                # put data on correct device
                X = X.to(self.device)
                
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output  = self.model(X)

                # calculate the loss
                loss = self.criterion(dec_output, X)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            # validate the model at the end of each epoch
            self.test_model(epoch, batch, valid_losses)

            # # update the learning rate scheduler
            # self.scheduler.step()
            # lr_train = self.scheduler.get_last_lr()[0]

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | train-loss: {train_loss:.4f}, "
                  f"validation-loss: {valid_loss:.4f}") # | lr: {lr_train:.6f}
            
            # print statistics in tensorboard
            self.writer.add_scalar("training-loss", train_loss, 
                                   epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("validation-loss", valid_loss, 
                                   epoch * len(self.train_loader) + batch)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        self.writer.flush()
        self.writer.close()

        fig = plot_losses(avg_train_losses, avg_valid_losses)
        self.writer.add_figure('loss-graph', fig)

        print("\nFinished training...\n")

    """ Method used to evaluate the model on the validation set. """
    def test_model(self, epoch, batch, test_losses):
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{self.num_epochs}]\n")
        
        # put model into evaluation mode
        self.model.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            test_loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True)
            
            for _, test_X in test_loop:
                test_X = test_X.to(self.device)
                
                test_enc_output, test_dec_output = self.model(test_X)

                test_loss = self.criterion(test_dec_output, test_X)

                test_losses.append(test_loss.item())

        # reput model into training mode
        self.model.train()
