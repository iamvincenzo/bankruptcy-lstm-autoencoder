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
            self.writer.add_scalar("only-ae-training-loss", train_loss,
                                   epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("only-ae-validation-loss", valid_loss,
                                   epoch * len(self.valid_loader) + batch)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.autoencoder)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        fig = plot_losses(avg_train_losses, avg_valid_losses)
        self.writer.add_figure('loss-graph', fig)

        self.writer.flush()
        self.writer.close()

        print("\nFinished autoencoder training...\n")

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

                _, dec_output = self.autoencoder(X)

                test_loss = self.criterion(X, dec_output)

                valid_losses.append(test_loss.item())

        # reput model into training mode
        self.autoencoder.train()

    # torch.set_printoptions(profile="full")

    """ Method used to compute the dot product between a matrix and a vector. """
    def compute_dot_product(self, mat, vec):    
        # mat-shape: [32, 5, 5] # vec-shape: [32, 5]        
        # expand dimensions of the vector to match the shape of the matrix
        exp_vec = vec.unsqueeze(2) # shape: [32, 5, 1]

        # perform the dot product using broadcasting: batched matrix x batched matrix
        result = torch.matmul(mat, exp_vec)  # shape: [32, 5, 1]

        # remove the extra dimension from the result
        result = result.squeeze(2)  # shape: [32, 5]

        return result

    """ Method used to train and validate the entire model with an early stopping implementation. """
    def train_all(self):
        print(f"\nStarting entire model training...")

        # to track the training loss as the model trains
        ae_train_losses, d90_train_losses, d5_train_losses  = [], [], []
        # to track the validation loss as the model trains
        ae_valid_losses, d90_valid_losses, d5_valid_losses  = [], [], []
        # to track the average training loss per epoch as the model trains
        ae_avg_train_losses, d90_avg_train_losses, d5_avg_train_losses = [], [], []
        # to track the average validation loss per epoch as the model trains
        ae_avg_valid_losses, d90_avg_valid_losses, d5_avg_valid_losses = [], [], []

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

            for batch, (ae_input, label) in loop:          
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(ae_input)

                # reshape the decoder output to match the input shape required by dense90-net
                d90_input = dec_output.reshape((dec_output.size(0),
                                                dec_output.size(2)*dec_output.size(1)))
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d90_output = self.dense90(d90_input)

                # compute the dense-net input
                d5_input = self.compute_dot_product(mat=enc_output, 
                                                    vec=d90_output)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input)

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(2)

                # calculate the loss
                ae_loss = self.criterion(ae_input, dec_output)
                # d90_loss = self.criterion1(d90_output, d90_output) # ??? quale è
                d5_loss = self.criterion2(label, d5_output)

                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss = ae_loss + d5_loss #+ d90_loss
                total_loss.backward()
                # ae_loss.backward(retain_graph=True)
                # d90_loss.backward()
                # d5_loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()
                self.optimizer1.step()
                self.optimizer2.step()

                # record training loss
                ae_train_losses.append(ae_loss.item())
                # d90_train_losses.append(d90_loss.item())
                d5_train_losses.append(d5_loss.item())

            # validate the model at the end of each epoch
            self.test_all(epoch, ae_valid_losses, d90_valid_losses, d5_valid_losses)

            # # update the learning rate scheduler
            # self.scheduler.step()
            # lr_train = self.scheduler.get_last_lr()[0]

            # calculate average loss over an epoch
            ae_train_loss = np.average(ae_train_losses)
            ae_valid_loss = np.average(ae_valid_losses)
            ae_avg_train_losses.append(ae_train_loss)
            ae_avg_valid_losses.append(ae_valid_loss)

            # d90_train_loss = np.average(d90_train_losses)
            # d90_valid_loss = np.average(d90_valid_losses)
            # d90_avg_train_losses.append(d90_train_loss)
            # d90_avg_valid_losses.append(d90_valid_loss)

            d5_train_loss = np.average(d5_train_losses)
            d5_valid_loss = np.average(d5_valid_losses)
            d5_avg_train_losses.append(d5_train_loss)
            d5_avg_valid_losses.append(d5_valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | "
                  f"ae_train-loss: {ae_train_loss:.4f}, ae_validation-loss: {ae_valid_loss:.4f} "
                #   f"d90_train-loss: {d90_train_loss:.4f}, d90_validation-loss: {d90_valid_loss:.4f} "
                  f"d5_train-loss: {d5_train_loss:.4f}, d5_validation-loss: {d5_valid_loss:.4f} ")  # | lr: {lr_train:.6f}

            # print statistics in tensorboard
            self.writer.add_scalar("ae-training-loss", ae_train_loss,
                                epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("ae-validation-loss", ae_valid_loss,
                                   epoch * len(self.valid_loader) + batch)
            # self.writer.add_scalar("d90-training-loss", d90_train_loss,
            #                     epoch * len(self.train_loader) + batch)
            # self.writer.add_scalar("d90-validation-loss", d90_valid_loss,
            #                        epoch * len(self.valid_loader) + batch)
            self.writer.add_scalar("d5-training-loss", d5_train_loss,
                                epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("d5-validation-loss", d5_valid_loss,
                                   epoch * len(self.valid_loader) + batch)
            
            total_train_loss = ae_train_loss + d5_train_loss # + d90_train_loss
            total_valid_loss = ae_valid_loss + d5_valid_loss # + d90_valid_loss
            self.writer.add_scalar("total-train-loss", total_train_loss,
                                epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("total-valid-loss", total_valid_loss,
                                   epoch * len(self.valid_loader) + batch)
            
            # clear lists to track next epoch
            ae_train_losses = []
            ae_valid_losses = []
            # d90_train_losses = []
            # d90_valid_losses = []
            d5_train_losses = []
            d5_valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            tot_loss = ae_valid_loss + d5_valid_loss # + d90_valid_loss
            early_stopping(tot_loss, self.autoencoder)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        fig = plot_losses(ae_avg_train_losses, ae_avg_valid_losses)
        self.writer.add_figure('ae-loss-graph', fig)

        # fig = plot_losses(d90_avg_train_losses, d90_avg_valid_losses)
        # self.writer.add_figure('d90-loss-graph', fig)

        fig = plot_losses(d5_avg_train_losses, d5_avg_valid_losses)
        self.writer.add_figure('d5-loss-graph', fig)

        self.writer.flush()
        self.writer.close()

        print("\nFinished entire model training...\n")

    """ Method used to compute the precision and recall in the validation set. """
    def compute_accuracy_precision_recall(self, epoch, predictions, targets):
        # Compute true positives, false positives, and false negatives
        predicted_classes = torch.argmax(predictions, dim=1)
        target_classes = torch.argmax(targets, dim=1)

        correct_predictions = (predicted_classes == target_classes).sum().item()
        true_positives = torch.sum((predicted_classes == 1) & (target_classes == 1)).item()
        false_positives = torch.sum((predicted_classes == 1) & (target_classes == 0)).item()
        false_negatives = torch.sum((predicted_classes == 0) & (target_classes == 1)).item()

        accuracy = (correct_predictions / targets.size(0))
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        # print statistics in tensorboard
        self.writer.add_scalar("valid-accuracy", accuracy,
                               epoch * len(self.valid_loader) + targets.size(0))
        self.writer.add_scalar("valid-precision", precision,
                               epoch * len(self.valid_loader) + targets.size(0))
        self.writer.add_scalar("valid-recall", recall,
                               epoch * len(self.valid_loader) + targets.size(0))

        print(f"\nEpoch [{epoch+1}/{self.num_epochs}] | Accuracy: {accuracy:.3f}, "
              f"Precision: {precision:.3f}, Recall: {recall:.3f}")

    """ Method used to validate the entire model. """
    def test_all(self, epoch, ae_valid_losses, d90_valid_losses, d5_valid_losses):
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{self.num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()
        self.dense90.eval()
        self.dense5.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True)

            total_predictions = []
            total_targets = []

            for _, (ae_input, label) in loop:
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(ae_input)

                # reshape the decoder output to match the input shape required by dense90-net
                d90_input = dec_output.reshape((dec_output.size(0),
                                                dec_output.size(2)*dec_output.size(1)))
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d90_output = self.dense90(d90_input)

                # compute the dense-net input
                d5_input = self.compute_dot_product(mat=enc_output, 
                                                    vec=d90_output)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input)

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(2)

                # calculate the loss
                ae_loss = self.criterion(ae_input, dec_output)
                # d90_loss = self.criterion1() ??? quale è
                d5_loss = self.criterion2(label, d5_output)

                # record training loss
                ae_valid_losses.append(ae_loss.item())
                # d90_valid_losses.append(d9_loss.item())
                d5_valid_losses.append(d5_loss.item())

                # append predictions and targets for computing epoch accuracy
                total_predictions.append(d5_output)
                total_targets.append(label)
            
            # concatenate all predictions and targets
            all_predictions = torch.cat(total_predictions)
            all_targets = torch.cat(total_targets)
            
            # compute the accuracy, precision and recall
            self.compute_accuracy_precision_recall(epoch=epoch, 
                                                   predictions=all_predictions,
                                                   targets=all_targets)

        # reput model into training mode
        self.autoencoder.train()
        self.dense90.train()
        self.dense5.train()
