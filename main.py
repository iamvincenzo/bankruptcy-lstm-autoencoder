import os
import torch
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from custom_dataset import get_data
from models import DenseSoftmaxLayer
from models import EncoderDecoderLSTM
from custom_dataset import CustomDataset


""" Function used to get command line parameters. """
def get_args():
    parser = ArgumentParser()

    # data-parameters
    #######################################################################################
    parser.add_argument("--data_path", type=str, default="./data",
                        help="path were to get the raw-dataset")
    parser.add_argument("--seq_len", type=int, default=5,
                        help="the lenght of the time-series sequence")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="the number of classes to predict")
    #######################################################################################

    # model-to-train
    #######################################################################################
    parser.add_argument("--train_only_ae", action="store_true",
                        help="train only the autoencoder model on alive companies")
    #######################################################################################

    # model-infos
    #######################################################################################
    parser.add_argument("--run_name", type=str, default="run_0",
                        help="the name assigned to the current run")
    parser.add_argument("--model_name", type=str, default="alive_ae",
                        help="the name of the model to be saved or loaded")
    #######################################################################################

    # model-parameters
    #######################################################################################
    parser.add_argument("--enc_input_size", type=int, default=18,
                        help="the number of expected features in the input x")
    parser.add_argument("--enc_hidden_size", type=int, default=5,  # 18,
                        help="the number of features in the hidden state h of the encoder")
    parser.add_argument("--dec_input_size", type=int, default=5,  # 18,
                        help="the number of expected features in the output of the encoder")
    parser.add_argument("--dec_hidden_size", type=int, default=18,
                        help="the number of features in the hidden state h of the decoder")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="the number of recurrent layers")
    #######################################################################################

    # training-parameters (1)
    #######################################################################################
    parser.add_argument("--num_epochs", type=int, default=10, # 20,
                        help="the total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="the batch size for training and test data")
    parser.add_argument("--workers", type=int, default=2,
                        help="the number of workers in the data loader")
    #######################################################################################

    # training-parameters (2)
    #######################################################################################
    parser.add_argument("--random_seed", type=int, default=42,
                        help="the random seed used to ensure reproducibility")
    parser.add_argument("--lr", type=float, default=0.001,  # default=0.0001,
                        help="the learning rate for optimization")
    parser.add_argument("--loss", type=str, default="rec_loss",
                        choices=["rec_loss", "mse_loss"],
                        help="the loss function used for model optimization")
    parser.add_argument("--opt", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="the optimizer used for training")
    parser.add_argument("--patience", type=int, default=5,
                        help="the threshold for early stopping during training")
    #######################################################################################

    # training-parameters (3)
    #######################################################################################
    parser.add_argument("--resume_train", action="store_true",
                        help="determines whether to load the model from a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints",
                        help="the path to save the trained model")
    #######################################################################################

    #######################################################################################
    parser.add_argument("--weights_init", action="store_true",
                        help="determines whether to use weights initialization")
    #######################################################################################

    return parser.parse_args()


""" Function used to run the experiments. """
def main(args):
    # tensorboard specifications
    t_date = "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
    writer = SummaryWriter("./runs/" + args.run_name + t_date)

    # get the data as numpy arrays
    np_train, np_valid, np_test = get_data(data_path=args.data_path, seq_len=args.seq_len,
                                           train_only_ae=args.train_only_ae, verbose=False)

    # dataset creation
    train_dataset = CustomDataset(x=np_train, seq_len=args.seq_len)
    valid_dataset = CustomDataset(x=np_valid, seq_len=args.seq_len)
    test_dataset = CustomDataset(x=np_test, seq_len=args.seq_len)

    # dataloader creation
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # select the device on which to place tensors
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: \n{device}")

    # lstm in encoder-decoder configuration
    autoencoder = EncoderDecoderLSTM(enc_input_size=args.enc_input_size,
                                     dec_input_size=args.dec_input_size,
                                     enc_hidden_size=args.enc_hidden_size,
                                     dec_hidden_size=args.dec_hidden_size,
                                     num_layers=args.num_layers,
                                     device=device,
                                     weights_init=args.weights_init)

    x, _ = next(iter(train_loader))    
    input_dim = x.size(2) * x.size(1)
    output_dim = x.size(1)
    dense90 = DenseSoftmaxLayer(input_dim=input_dim,
                                output_dim=output_dim,
                                weights_init=args.weights_init)

    input_dim = x.size(1)
    output_dim = args.num_classes
    dense5 = DenseSoftmaxLayer(input_dim=input_dim,
                               output_dim=output_dim,
                               weights_init=args.weights_init)

    # solver for training, validation and test
    solver = Solver(models=(autoencoder, dense90, dense5),
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    writer=writer,
                    device=device,
                    args=args)

    if args.train_only_ae:
        solver.train_ae()
    else:
        solver.train_all()


""" Runs the simulation. """
if __name__ == "__main__":
    args = get_args()

    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    print(f"\n{args}")
    main(args)
