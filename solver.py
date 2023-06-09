import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from metrics import RECLoss
from metrics import compute_metrics
from pytorchtools import EarlyStopping
from pltotting_utils import plot_losses
from metrics import reconstruction_for_prior
from pltotting_utils import plot_confusion_matrix
from custom_dataset import get_valid_failed_dataloader


""" Class for training, validation and testing. """
class Solver(object):
    """ Initialize configuration. """
    def __init__(self, models, train_loader, valid_loader, test_loader, writer, device, args):
        super().__init__()
        self.args = args
        self.model_name = f"model_{self.args.model_name}.pt"

        # models to train/validate
        self.autoencoder, self.dense5 = models
        self.autoencoder = self.autoencoder.to(device)
        self.dense5 = self.dense5.to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # load a pretrained model
        if self.args.resume_train or not(self.args.train_only_ae):
            self.load_model(self.autoencoder, device)

        # select the loss for LSTM-AE optimization
        if self.args.loss == "rec_loss":
            self.ae_criterion = RECLoss()
        elif self.args.loss == "mse_loss":
            self.ae_criterion = nn.MSELoss()

        # select the loss for FC-Dense5 optimization
        self.d5_criterion = nn.CrossEntropyLoss()

        # choose the optimizer
        if self.args.opt == "Adam":
            self.ae_optimizer = optim.Adam(params=self.autoencoder.parameters(),
                                           lr=self.args.lr, betas=(0.9, 0.999))
            self.d5_optimizer = optim.Adam(params=self.dense5.parameters(),
                                           lr=self.args.lr, betas=(0.9, 0.999))
        elif self.args.opt == "SGD":
            self.ae_optimizer = optim.SGD(params=self.autoencoder.parameters(),
                                          lr=self.args.lr, momentum=0.9)            
            self.d5_optimizer = optim.SGD(params=self.dense5.parameters(),
                                          lr=0.001, momentum=0.9)

        # other training/validation params
        self.num_epochs = args.num_epochs
        self.patience = self.args.patience
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.device = device

        # # visualize the model we built on tensorboard
        # x, _ = next(iter(self.train_loader))
        # self.writer.add_graph(self.autoencoder, x.to(device))
        # self.writer.close()

    """ Method used to load a model. """
    def load_model(self, model, device):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        model.load_state_dict(torch.load(check_path, 
                                         map_location=torch.device(device=device)))
        print("\nModel loaded...")

    """ Method used to compute the batch dot product between a matrix and a vector. """
    def compute_dot_product(self, mat, vec):
        # mat-shape: [32, 5, 5] 
        # vec-shape: [32, 5]        
        # expand dimensions of the vector to match the shape of the matrix
        exp_vec = vec.unsqueeze(2) # shape: [32, 5, 1]

        # perform the dot product using broadcasting: batched matrix x batched matrix
        result = torch.matmul(mat, exp_vec)  # shape: [32, 5, 1]

        # remove the extra dimension from the result
        result = result.squeeze(2)  # shape: [32, 5]

        return result

    """" Method used to freeze the LSTM-AE weights. """
    def freeze_lstm_ae(self):
        for n, m in self.autoencoder.named_parameters():
            m.requires_grad = False
        
        # print for debugging
        for n, m in self.autoencoder.named_parameters():
            print(f"name: {n}, trainable: {m.requires_grad}")

    # METHODS USED BY CONFIGURATION_1 and CONFIGURATION_2
    ###################################################################################################
    """ Method used to train the autoencoder model with early stopping implementation. """
    def train_ae(self):
        print(f"\nStarting LSTM-Autoencoder training...")

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
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch_id, (ae_input, _) in loop:
                # put data on correct device
                ae_input = ae_input.to(self.device)

                # clear the gradients of all optimized variables
                self.ae_optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(ae_input)

                # calculate the loss
                loss = self.ae_criterion(y_pred=dec_output, y_true=ae_input)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.ae_optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            # validate the model at the end of each epoch
            self.test_ae(epoch=epoch, data_loader=self.valid_loader, 
                         valid_losses=valid_losses, valid=True)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | train-loss: {train_loss:.4f}, "
                  f"validation-loss: {valid_loss:.4f}")  

            # print statistics in tensorboard
            self.writer.add_scalar("only-ae-training-loss", train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("only-ae-validation-loss", valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)

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

        print("\nLSTM-Autoencoder training Done...\n")

        # INFERECE (VALIDATION-SET) with only failed companies
        ###########################################################################################
        print("\nStarting the inference on the validation-set with only failed companies...")

        valid_failed_loader = get_valid_failed_dataloader(self.args.data_path,
                                                          self.args.seq_len,
                                                          self.args.batch_size,
                                                          self.args.workers)
        epoch=0
        valid_failed_losses = []
        # validate the model on failed companies
        self.test_ae(epoch=epoch, data_loader=valid_failed_loader,
                     valid_losses=valid_failed_losses, valid=False)

        # calculate average loss over an epoch
        valid_failed_loss = np.average(valid_failed_losses)

        # print some statistics
        print(f"\nTest[{epoch + 1}/{epoch + 1}] | "
              f"mean-reconstruction-error on failed companies: {valid_failed_loss}")  

        print("\nInference on the validation-set with only failed companies Done...\n")
        ###########################################################################################

    """ Method used to evaluate the model on the validation set. """
    def test_ae(self, epoch, data_loader, valid_losses, valid=True):
        if valid:
            num_epochs = self.num_epochs
        else:
            num_epochs = epoch + 1

        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

            for _, (ae_input, _) in loop:
                # put data on correct device
                ae_input = ae_input.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                _, dec_output = self.autoencoder(ae_input)

                # calculate the loss
                test_loss = self.ae_criterion(y_pred=dec_output, y_true=ae_input)

                # record training loss
                valid_losses.append(test_loss.item())

        # reput model into training mode
        self.autoencoder.train()
        
    """ Method used to train and validate the entire model with an early stopping implementation. """
    def train_all(self):
        print(f"\nStarting (LSTM-AE + FC-Dense5)-model training...")

        # do not overwrite aeutoencoder trained model
        self.args.model_name = self.args.model_name + "_classif_all"
        self.model_name = f"model_{self.args.model_name}.pt"

        # to track the training loss as the model trains
        ae_train_losses, d5_train_losses  = [], []
        # to track the validation loss as the model trains
        ae_valid_losses, d5_valid_losses  = [], []
        # to track the average training loss per epoch as the model trains
        ae_avg_train_losses, d5_avg_train_losses = [], []
        # to track the average validation loss per epoch as the model trains
        ae_avg_valid_losses, d5_avg_valid_losses = [], []

        # initialize the early_stopping object
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        early_stopping = EarlyStopping(patience=self.patience,
                                       verbose=True, path=check_path)
        
        # lstm-autoencoder is not 
        # trained during the classification task
        if self.args.freeze_ae:
            print(f"\nFreezing LSTM-AE weights...")
            self.freeze_lstm_ae()

        # put the models in training mode
        self.autoencoder.train()
        self.dense5.train()

        # loop over the dataset multiple times
        for epoch in range(self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]\n")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch_id, (ae_input, label) in loop:          
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # clear the gradients of all optimized variables
                self.ae_optimizer.zero_grad()
                self.d5_optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(ae_input)

                # compute the d5 input differently for config2
                if self.args.config_2:
                    # shape[32, 25]
                    d5_input = enc_output.reshape(enc_output.size(0), 
                                                  enc_output.size(1) * enc_output.size(1))
                else:
                    # compute softmax-input
                    rec_prior = reconstruction_for_prior(y_pred=dec_output,
                                                         y_true=ae_input)

                    # compute softmax-output used to obtain d5_input
                    rec_prior_prob = self.softmax(rec_prior)
                
                    # shape[32, 5]
                    # compute the fc-dense5-net input: combine 'enc_output' with 'rec_prior_prob'
                    d5_input = self.compute_dot_product(mat=enc_output, 
                                                        vec=rec_prior_prob)             
                      
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input) # shape [32, 2]

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(1).long() # shape [32, 1] --> [32]               

                # calculate the loss
                ae_loss = self.ae_criterion(y_pred=dec_output, y_true=ae_input)
                d5_loss = self.d5_criterion(input=d5_output, target=label)

                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss = ae_loss + d5_loss
                total_loss.backward()
                # ae_loss.backward(retain_graph=True)
                # d5_loss.backward()

                # perform a single optimization step (parameter update)
                self.ae_optimizer.step()
                self.d5_optimizer.step()

                # record training loss
                ae_train_losses.append(ae_loss.item())
                d5_train_losses.append(d5_loss.item())

            # validate the model at the end of each epoch
            self.test_all(epoch=epoch, data_loader=self.valid_loader, 
                          ae_losses=ae_valid_losses, d5_losses=d5_valid_losses, valid=True)

            # calculate average loss over an epoch
            ae_train_loss = np.average(ae_train_losses)
            ae_valid_loss = np.average(ae_valid_losses)
            ae_avg_train_losses.append(ae_train_loss)
            ae_avg_valid_losses.append(ae_valid_loss)

            d5_train_loss = np.average(d5_train_losses)
            d5_valid_loss = np.average(d5_valid_losses)
            d5_avg_train_losses.append(d5_train_loss)
            d5_avg_valid_losses.append(d5_valid_loss)

            total_train_loss = ae_train_loss + d5_train_loss
            total_valid_loss = ae_valid_loss + d5_valid_loss

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | "
                  f"ae_train-loss: {ae_train_loss:.4f}, ae_validation-loss: {ae_valid_loss:.4f} "
                  f"d5_train-loss: {d5_train_loss:.4f}, d5_validation-loss: {d5_valid_loss:.4f} "
                  f"tot-train_loss: {total_train_loss:.4f}, tot-valid_loss: {total_valid_loss:.4f} ") 

            # print statistics in tensorboard
            self.writer.add_scalar("ae-training-loss", ae_train_loss, 
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("ae-validation-loss", ae_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)
            self.writer.add_scalar("d5-training-loss", d5_train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("d5-validation-loss", d5_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)           
            self.writer.add_scalar("total-train-loss", total_train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("total-valid-loss", total_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)
            
            # clear lists to track next epoch
            ae_train_losses = []
            ae_valid_losses = []
            d5_train_losses = []
            d5_valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            tot_loss = ae_valid_loss + d5_valid_loss
            early_stopping(tot_loss, self.autoencoder)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        fig = plot_losses(ae_avg_train_losses, ae_avg_valid_losses)
        self.writer.add_figure('ae-loss-graph', fig)

        fig = plot_losses(d5_avg_train_losses, d5_avg_valid_losses)
        self.writer.add_figure('d5-loss-graph', fig)

        print("\n(LSTM-AE + FC-Dense5)-model training Done...\n")

        # INFERECE (TEST-SET)
        ###########################################################################################
        print("\nStarting the inference on the test set...")

        epoch=0
        ae_test_losses, d5_test_losses = [], []

        # test the model
        self.test_all(epoch=epoch, data_loader=self.test_loader, 
                      ae_losses=ae_test_losses, d5_losses=d5_test_losses, valid=False)

        # calculate average loss over an epoch
        ae_test_loss = np.average(ae_test_losses)
        d5_test_loss = np.average(d5_test_losses)

        # print some statistics
        print(f"\nTest[{epoch + 1}/{epoch + 1}] | ae_test-loss: {ae_test_loss:.4f} "
              f"d5_test-loss: {d5_test_loss:.4f} ")  

        print("\nInference on the test set Done...\n")

        self.writer.flush()
        self.writer.close()
        ###########################################################################################

    """ Method used to validate the entire model. """
    def test_all(self, epoch, data_loader, ae_losses, d5_losses, valid=True):
        if valid:
            num_epochs = self.num_epochs
        else:
            num_epochs = epoch + 1
        
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()
        self.dense5.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

            total_predictions = []
            total_targets = []

            for _, (ae_input, label) in loop:
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, dec_output = self.autoencoder(ae_input)

                # compute the d5 input differently for config2
                if self.args.config_2:
                    # shape[32, 25]
                    d5_input = enc_output.reshape(enc_output.size(0), 
                                                  enc_output.size(1) * enc_output.size(1))
                else:
                    # compute softmax-input
                    rec_prior = reconstruction_for_prior(y_pred=dec_output,
                                                         y_true=ae_input)

                    # compute softmax-output used to obtain d5_input
                    rec_prior_prob = self.softmax(rec_prior)
                
                    # shape[32, 5]
                    # compute the fc-dense5-net input: combine 'enc_output' with 'rec_prior_prob'
                    d5_input = self.compute_dot_product(mat=enc_output, 
                                                        vec=rec_prior_prob)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input)

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(1).long() # shape [32, 1] --> [32]

                # calculate the loss
                ae_loss = self.ae_criterion(y_pred=dec_output, y_true=ae_input)
                d5_loss = self.d5_criterion(input=d5_output, target=label)

                # record training loss
                ae_losses.append(ae_loss.item())
                d5_losses.append(d5_loss.item())

                # append predictions and targets for computing epoch accuracy
                total_predictions.append(d5_output)
                total_targets.append(label)
            
            # concatenate all predictions and targets along 0-axis (vertically)
            all_predictions = torch.cat(total_predictions, dim=0)
            all_targets = torch.cat(total_targets, dim=0)

            from torch.nn import Softmax
            softmax = Softmax(dim=1)

            # compute some metrics
            (accuracy, precision, recall, f1_score, 
             specificity, conf_matr) = compute_metrics(predictions=softmax(all_predictions),
                                                       targets=all_targets)            

            print(f"\nEpoch [{epoch+1}/{num_epochs}] | Accuracy: {accuracy:.3f}, "
                  f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}, "
                  f"Specificity: {specificity:.3f}")

            print(f"\nConfusion_matrix: \n{conf_matr}")

            if valid:
                # print statistics in tensorboard
                self.writer.add_scalar("validation-precision", precision,
                                       epoch * len(data_loader)) # + batch)
                self.writer.add_scalar("validation-recall", recall,
                                       epoch * len(data_loader)) # + batch)
                self.writer.add_scalar("validation-f1_score", f1_score,
                                       epoch * len(self.train_loader)) # + batch)
                self.writer.add_scalar("d5-specificity", specificity,
                                       epoch * len(data_loader)) # + batch)

            confusion_matrix_np = conf_matr.numpy()
            tn, fp, fn, tp = confusion_matrix_np.ravel()

            if valid:
                text = "confusion-matrix-valid"
            else:
                text = "confusion-matrix-test"

            fig = plot_confusion_matrix(tn=tn, fp=fp, fn=fn, tp=tp)
            self.writer.add_figure(text, fig, global_step=epoch * len(self.train_loader)) # + batch)         

        # reput model into training mode
        self.autoencoder.train()
        self.dense5.train()
    ###################################################################################################

    # METHODS USED BY CONFIGURATION_3
    ###################################################################################################
    """ Method used to train the autoencoder with attention model with early stopping implementation. """
    def train_ae_luong_att(self):
        print(f"\nStarting LSTM-Autoencoder training with Luong Attention...")

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
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch_id, (ae_input, _) in loop:
                # put data on correct device
                ae_input = ae_input.to(self.device)

                # clear the gradients of all optimized variables
                self.ae_optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, _, output = self.autoencoder(ae_input)

                # calculate the loss
                loss = self.ae_criterion(y_pred=output, y_true=ae_input)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                
                # perform a single optimization step (parameter update)
                self.ae_optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            # validate the model at the end of each epoch
            self.test_ae_luong_att(epoch=epoch, data_loader=self.valid_loader,
                                   valid_losses=valid_losses, valid=True)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | train-loss: {train_loss:.4f}, "
                  f"validation-loss: {valid_loss:.4f}")  

            # print statistics in tensorboard
            self.writer.add_scalar("only-ae-training-loss", train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("only-ae-validation-loss", valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)

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

        print(f"\nLSTM-Autoencoder training with Luong Attention Done...")

        # INFERECE (VALIDATION-SET) with only failed companies
        ###########################################################################################
        print("\nStarting the inference on the validation-set with only failed companies...")

        valid_failed_loader = get_valid_failed_dataloader(self.args.data_path,
                                                          self.args.seq_len,
                                                          self.args.batch_size,
                                                          self.args.workers)
        epoch=0
        valid_failed_losses = []
        # validate the model on failed companies
        self.test_ae_luong_att(epoch=epoch, data_loader=valid_failed_loader,
                               valid_losses=valid_failed_losses, valid=False)

        # calculate average loss over an epoch
        valid_failed_loss = np.average(valid_failed_losses)

        # print some statistics
        print(f"\nTest[{epoch + 1}/{epoch + 1}] | "
              f"mean-reconstruction-error on failed companies: {valid_failed_loss}")  

        print("\nInference on the validation-set with only failed companies Done...\n")
        ###########################################################################################

    """ Method used to evaluate the model with luong attention on the validation set. """
    def test_ae_luong_att(self, epoch, data_loader, valid_losses, valid=True):
        if valid:
            num_epochs = self.num_epochs
        else:
            num_epochs = epoch + 1

        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

            for _, (ae_input, _) in loop:
                # put data on correct device
                ae_input = ae_input.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, _, output = self.autoencoder(ae_input)

                # calculate the loss
                test_loss = self.ae_criterion(y_pred=output, y_true=ae_input)
                
                # record training loss
                valid_losses.append(test_loss.item())

        # reput model into training mode
        self.autoencoder.train()
    
    """ Method used to train and validate the entire model with an early stopping implementation. """
    def train_all_ae_luong_att(self):
        print(f"\nStarting (LSTM-AE with Att. + FC-Dense5)-model training...")

        # do not overwrite aeutoencoder trained model
        self.args.model_name = self.args.model_name + "_classif_all"
        self.model_name = f"model_{self.args.model_name}.pt"

        # to track the training loss as the model trains
        ae_train_losses, d5_train_losses  = [], []
        # to track the validation loss as the model trains
        ae_valid_losses, d5_valid_losses  = [], []
        # to track the average training loss per epoch as the model trains
        ae_avg_train_losses, d5_avg_train_losses = [], []
        # to track the average validation loss per epoch as the model trains
        ae_avg_valid_losses, d5_avg_valid_losses = [], []

        # initialize the early_stopping object
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        early_stopping = EarlyStopping(patience=self.patience,
                                       verbose=True, path=check_path)
        
        # lstm-autoencoder is not 
        # trained during the classification task
        if self.args.freeze_ae:
            print(f"\nFreezing LSTM-AE + Luong Att. weights...")
            self.freeze_lstm_ae()

        # put the models in training mode
        self.autoencoder.train()
        self.dense5.train()

        # loop over the dataset multiple times
        for epoch in range(self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]\n")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch_id, (ae_input, label) in loop:          
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # clear the gradients of all optimized variables
                self.ae_optimizer.zero_grad()
                self.d5_optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, _, output = self.autoencoder(ae_input)

                # compute softmax-input
                rec_prior = reconstruction_for_prior(y_pred=output, y_true=ae_input)

                # compute softmax-output used to obtain d5_input
                rec_prior_prob = self.softmax(rec_prior)
            
                # shape[32, 5]
                # compute the fc-dense5-net input: combine 'enc_output' with 'rec_prior_prob'
                d5_input = self.compute_dot_product(mat=enc_output, vec=rec_prior_prob)             
                      
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input) # shape [32, 2]

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(1).long() # shape [32, 1] --> [32]               

                # calculate the loss
                ae_loss = self.ae_criterion(y_pred=output, y_true=ae_input)
                d5_loss = self.d5_criterion(input=d5_output, target=label)

                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss = ae_loss + d5_loss
                total_loss.backward()
                # ae_loss.backward(retain_graph=True)
                # d5_loss.backward()

                # perform a single optimization step (parameter update)
                self.ae_optimizer.step()
                self.d5_optimizer.step()

                # record training loss
                ae_train_losses.append(ae_loss.item())
                d5_train_losses.append(d5_loss.item())

            # validate the model at the end of each epoch
            self.test_all_ae_luong(epoch=epoch, data_loader=self.valid_loader, 
                                   ae_losses=ae_valid_losses, d5_losses=d5_valid_losses, valid=True)

            # calculate average loss over an epoch
            ae_train_loss = np.average(ae_train_losses)
            ae_valid_loss = np.average(ae_valid_losses)
            ae_avg_train_losses.append(ae_train_loss)
            ae_avg_valid_losses.append(ae_valid_loss)

            d5_train_loss = np.average(d5_train_losses)
            d5_valid_loss = np.average(d5_valid_losses)
            d5_avg_train_losses.append(d5_train_loss)
            d5_avg_valid_losses.append(d5_valid_loss)

            total_train_loss = ae_train_loss + d5_train_loss
            total_valid_loss = ae_valid_loss + d5_valid_loss

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | "
                  f"ae_train-loss: {ae_train_loss:.4f}, ae_validation-loss: {ae_valid_loss:.4f} "
                  f"d5_train-loss: {d5_train_loss:.4f}, d5_validation-loss: {d5_valid_loss:.4f} "
                  f"tot-train_loss: {total_train_loss:.4f}, tot-valid_loss: {total_valid_loss:.4f} ") 

            # print statistics in tensorboard
            self.writer.add_scalar("ae-training-loss", ae_train_loss, 
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("ae-validation-loss", ae_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)
            self.writer.add_scalar("d5-training-loss", d5_train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("d5-validation-loss", d5_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)           
            self.writer.add_scalar("total-train-loss", total_train_loss,
                                   epoch * len(self.train_loader)) # + batch)
            self.writer.add_scalar("total-valid-loss", total_valid_loss,
                                   epoch * len(self.valid_loader)) # + batch)
            
            # clear lists to track next epoch
            ae_train_losses = []
            ae_valid_losses = []
            d5_train_losses = []
            d5_valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            tot_loss = ae_valid_loss + d5_valid_loss
            early_stopping(tot_loss, self.autoencoder)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        fig = plot_losses(ae_avg_train_losses, ae_avg_valid_losses)
        self.writer.add_figure('ae-loss-graph', fig)

        fig = plot_losses(d5_avg_train_losses, d5_avg_valid_losses)
        self.writer.add_figure('d5-loss-graph', fig)

        print("\n(LSTM-AE with Att. + FC-Dense5)-model training Done...\n")

        # INFERECE (TEST-SET)
        ###########################################################################################
        print("\nStarting the inference on the test set...")

        epoch=0
        ae_test_losses, d5_test_losses = [], []

        # test the model
        self.test_all_ae_luong(epoch=epoch, data_loader=self.test_loader, 
                               ae_losses=ae_test_losses, d5_losses=d5_test_losses, valid=False)

        # calculate average loss over an epoch
        ae_test_loss = np.average(ae_test_losses)
        d5_test_loss = np.average(d5_test_losses)

        # print some statistics
        print(f"\nTest[{epoch + 1}/{epoch + 1}] | ae_test-loss: {ae_test_loss:.4f} "
              f"d5_test-loss: {d5_test_loss:.4f} ")  

        print("\nInference on the test set Done...\n")

        self.writer.flush()
        self.writer.close()
        ###########################################################################################

    """ Method used to validate the entire model. """
    def test_all_ae_luong(self, epoch, data_loader, ae_losses, d5_losses, valid=True):
        if valid:
            num_epochs = self.num_epochs
        else:
            num_epochs = epoch + 1
        
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{num_epochs}]\n")

        # put model into evaluation mode
        self.autoencoder.eval()
        self.dense5.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

            total_predictions = []
            total_targets = []

            for _, (ae_input, label) in loop:
                # put data on correct device
                ae_input, label = ae_input.to(self.device), label.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                enc_output, _, output = self.autoencoder(ae_input)
                
                # compute softmax-input
                rec_prior = reconstruction_for_prior(y_pred=output, y_true=ae_input)

                # compute softmax-output used to obtain d5_input
                rec_prior_prob = self.softmax(rec_prior)
            
                # shape[32, 5]
                # compute the fc-dense5-net input: combine 'enc_output' with 'rec_prior_prob'
                d5_input = self.compute_dot_product(mat=enc_output, vec=rec_prior_prob)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                d5_output = self.dense5(d5_input)

                # reshape label to match the output shape of dense5-net
                label = label.squeeze(1).long() # shape [32, 1] --> [32]

                # calculate the loss
                ae_loss = self.ae_criterion(y_pred=output, y_true=ae_input)
                d5_loss = self.d5_criterion(input=d5_output, target=label)

                # record training loss
                ae_losses.append(ae_loss.item())
                d5_losses.append(d5_loss.item())

                # append predictions and targets for computing epoch accuracy
                total_predictions.append(d5_output)
                total_targets.append(label)
            
            # concatenate all predictions and targets along 0-axis (vertically)
            all_predictions = torch.cat(total_predictions, dim=0)
            all_targets = torch.cat(total_targets, dim=0)

            from torch.nn import Softmax
            softmax = Softmax(dim=1)

            # compute some metrics
            (accuracy, precision, recall, f1_score, 
             specificity, conf_matr) = compute_metrics(predictions=softmax(all_predictions),
                                                       targets=all_targets)            

            print(f"\nEpoch [{epoch+1}/{num_epochs}] | Accuracy: {accuracy:.3f}, "
                  f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}, "
                  f"Specificity: {specificity:.3f}")

            print(f"\nConfusion_matrix: \n{conf_matr}")

            if valid:
                # print statistics in tensorboard
                self.writer.add_scalar("validation-precision", precision,
                                       epoch * len(data_loader)) # + batch)
                self.writer.add_scalar("validation-recall", recall,
                                       epoch * len(data_loader)) # + batch)
                self.writer.add_scalar("validation-f1_score", f1_score,
                                       epoch * len(self.train_loader)) # + batch)
                self.writer.add_scalar("d5-specificity", specificity,
                                       epoch * len(data_loader)) # + batch)

            confusion_matrix_np = conf_matr.numpy()
            tn, fp, fn, tp = confusion_matrix_np.ravel()

            if valid:
                text = "confusion-matrix-valid"
            else:
                text = "confusion-matrix-test"

            fig = plot_confusion_matrix(tn=tn, fp=fp, fn=fn, tp=tp)
            self.writer.add_figure(text, fig, global_step=epoch * len(self.train_loader)) # + batch)         

        # reput model into training mode
        self.autoencoder.train()
        self.dense5.train()
    ###################################################################################################