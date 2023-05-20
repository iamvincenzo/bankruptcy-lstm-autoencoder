import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from metrics import Rec_Loss
from pytorchtools import EarlyStopping
from pltotting_utils import plot_losses



""" Class for training, validation and testing. """
class Solver(object):
    """ Initialize configuration. """
    def __init__(self, models, train_loader, valid_loader, test_loader, writer, device, args):
        super().__init__()
        self.args = args
        self.model_name = f"lstm_encdec_{self.args.model_name}.pt"

        # models to train/validate
        self.autoencoder, self.dense90, self.dense5 = models
        self.autoencoder = self.autoencoder.to(device)
        self.dense90 = self.dense90.to(device)
        self.dense5 = self.dense5.to(device)

        # load a pretrained model
        if self.args.resume_train or not(self.args.train_only_ae):
            self.load_model(self.autoencoder, device)

        # select the loss for optimization
        if self.args.loss == "rec_loss":
            self.criterion = Rec_Loss()
        elif self.args.loss == "mse_loss":
            self.criterion = nn.MSELoss()

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()

        # choose the optimizer
        if self.args.opt == "Adam":
            self.optimizer = optim.Adam(params=self.autoencoder.parameters(),
                                        lr=self.args.lr, betas=(0.9, 0.999))
        elif self.args.opt == "SGD":
            self.optimizer = optim.SGD(params=self.autoencoder.parameters(),
                                       lr=self.args.lr, momentum=0.9)
            
        self.optimizer1 = optim.SGD(params=self.dense90.parameters(), 
                                    lr=0.001, momentum=0.9)
        self.optimizer2 = optim.SGD(params=self.dense5.parameters(), 
                                    lr=0.001, momentum=0.9)

        # other training/validation params
        self.num_epochs = args.num_epochs
        self.patience = self.args.patience
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.device = device

        # visualize the model we built on tensorboard
        x, _ = next(iter(self.train_loader))
        self.writer.add_graph(self.autoencoder, x.to(device))
        self.writer.close()        

    """ Method used to load a model. """
    def load_model(self, model, device):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        model.load_state_dict(torch.load(check_path, 
                                         map_location=torch.device(device=device)))
        print("\nModel loaded...")

    """ Method used to train the autoencoder model with early stopping implementation. """
    def train_ae(self):
        print(f"\nStarting autoencoder training...")

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
        self.autoencoder.train()

        # loop over the dataset multiple times
        for epoch in range(self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]\n")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader), leave=True)

            for batch, (X, _) in loop:
                # put data on correct device
                X = X.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(X)

                # calculate the loss
                loss = self.criterion(X, dec_output)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            # validate the model at the end of each epoch
            self.test_ae(epoch, valid_losses)

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
                  f"validation-loss: {valid_loss:.4f}")  # | lr: {lr_train:.6f}

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
            early_stopping(valid_loss, self.autoencoder)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        self.writer.flush()
        self.writer.close()

        fig = plot_losses(avg_train_losses, avg_valid_losses)
        self.writer.add_figure('loss-graph', fig)

        print("\nFinished training...\n")

    """ Method used to evaluate the model on the validation set. """
    def test_ae(self, epoch, valid_losses):
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{self.num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True)

            for _, (X, _) in loop:
                X = X.to(self.device)

                enc_output, dec_output = self.autoencoder(X)

                test_loss = self.criterion(X, dec_output)

                valid_losses.append(test_loss.item())

        # reput model into training mode
        self.autoencoder.train()

    if True:
        def compute_dot_product(self, mat1, mat2):
            # # Create the input tensors
            # matrix = torch.rand(32, 5, 5)  # Shape: [32, 5, 5]
            # vector = torch.rand(32, 5)  # Shape: [32, 5]

            # # Expand dimensions of the vector to match the shape of the matrix
            # expanded_vector = vector.unsqueeze(2)  # Shape: [32, 5, 1]

            # Perform the dot product using broadcasting: batched matrix x batched matrix
            result = torch.matmul(mat1, mat2)  # Shape: [32, 5, 1]

            # Remove the extra dimension from the result
            result = result.squeeze(2)  # Shape: [32, 5]

            # print(result.shape)  # Shape of the resulting tensor

            return result

        # def train_entire_model(self):
        #     num_epochs = 10
        #     for epoch in range(num_epochs):
        #         running_loss = 0.0

        #         # Forward pass
        #         outputs = self.dense90(inputs)
        #         loss = self.criterion2(outputs, labels)

        #         # Backward pass and optimization
        #         self.optimizer1.zero_grad()
        #         self.loss1.backward()
        #         self.optimizer1.step()

        #         # Print statistics
        #         running_loss += loss.item()
        #         if (epoch + 1) % 10 == 0:
        #             print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

        #     print("Training finished.")

        def train_all(self):
            print(f"\nStarting autoencoder training...")

            self.load_model(self.autoencoder, self.device)

            # to track the training loss as the model trains
            autoenc_train_losses, dense90_train_losses, dense5_train_losses  = [], [], []
            # to track the validation loss as the model trains
            autoenc_valid_losses, dense90_valid_losses, dense5_valid_losses  = [], [], []
            # to track the average training loss per epoch as the model trains
            autoenc_avg_train_losses, dense90_avg_train_losses, dense5_avg_train_losses = [], [], []
            # to track the average validation loss per epoch as the model trains
            autoenc_avg_valid_losses, dense90_avg_valid_losses, dense5_avg_valid_losses = [], [], []

            # initialize the early_stopping object
            check_path = os.path.join(self.args.checkpoint_path, self.model_name)
            early_stopping = EarlyStopping(patience=self.patience,
                                           verbose=True, path=check_path)

            # put the models in training mode
            self.autoencoder.train()
            self.dense90.train()
            self.dense5.train()

            # loop over the dataset multiple times
            for epoch in range(self.num_epochs):
                print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]\n")

                # used for creating a terminal progress bar
                loop = tqdm(enumerate(self.train_loader),
                            total=len(self.train_loader), leave=True)

                for batch, X in loop:
                    pass
                    # x = x.reshape((x.size(0), x.size(2) * x.size(1)))

        def test_all(self):
            pass
