##########################################################################################################
# SOME REFERENCES: https://medium.com/towards-data-science/whats-happening-in-my-lstm-layer-dd8110ecc52f #
##########################################################################################################

#################################################################################
# SOME REFERENCES: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html #
#################################################################################

# For each layer in your LSTM — the number of cells is equal to the size of your window.

import torch
import torch.nn as nn

""" Class form LSTM in Autoencoder configuration. """
class EncoderDecoderLSTM(nn.Module):
    """ Initialize configurations. """
    def __init__(self, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, num_layers, device, bidirectional=False):
        super(EncoderDecoderLSTM, self).__init__()
        # the number of expected features in the input x
        self.enc_input_size = enc_input_size
        # the number of expected features in the output of the encoder
        self.dec_input_size = dec_input_size
        # the number of features in the hidden state h of the encoder
        self.enc_hidden_size = enc_hidden_size
        # the number of features in the hidden state h of the decoder
        self.dec_hidden_size = dec_hidden_size
        # number of recurrent layers
        self.num_layers = num_layers
        # if true becomes a bidirectional LSTM
        D = 2 if bidirectional else 1
        self.directions = D
        # device where to put tensors
        self.device = device

        # lstm-architecture
        self.lstm_encoder = nn.LSTM(input_size=enc_input_size,
                                    hidden_size=enc_hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        # lstm-architecture
        self.lstm_decoder = nn.LSTM(input_size=dec_input_size,
                                    hidden_size=dec_hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        # self._reinitialize()

    """ Tensorflow/Keras-like initialization. """
    def _reinitialize(self):    
        print('\nPerforming weights initialization...')

        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    """ Method used to define the forward pass of the input through the network during the training. """
    def forward(self, x):
        # # ENCODER 
        """ input shape (batch_size, sequence_length, number_features) when batch_first=True """
        # print(f"\ninput-shape: \n{x.shape}")
        # print(f"\ninput: \n{x}")

        batch_size = x.size(0)
        """ (D ∗ num_layers, batch_size, hidden_size) """
        h_0 = torch.zeros((self.directions * self.num_layers,
                           batch_size, self.enc_hidden_size)).to(self.device)
        c_0 = torch.zeros((self.directions * self.num_layers,
                           batch_size, self.enc_hidden_size)).to(self.device)

        """ output-shape (batch_size, sequence_lenght, D * hidden_size)
        h_n-shape (D * num_layers, batch_size, hidden_size)
        c_n-shape (D * num_layers, batch_size, hidden_size) """
        enc_output, (h_n, c_n) = self.lstm_encoder(x, (h_0, c_0))

        # print(f"\nh_n-shape: \n{h_n.shape}")
        # print(f"\nh_n: \n{h_n}")
        # print(f"\nc_n-shape: \n{c_n.shape}")
        # print(f"\nc_n: \n{c_n}")

        # # DECODER
        # print(f"\nenc-output-shape: \n{enc_output.shape}")
        # print(f"\nenc-output: \n{enc_output}")

        batch_size = enc_output.size(0)
        """ (D ∗ num_layers, batch_size, hidden_size) """
        h_0 = torch.zeros((self.directions * self.num_layers,
                           batch_size, self.dec_hidden_size)).to(self.device)
        c_0 = torch.zeros((self.directions * self.num_layers,
                           batch_size, self.dec_hidden_size)).to(self.device)

        """ output-shape (batch_size, sequence_lenght, D * hidden_size)
        h_n-shape (D * num_layers, batch_size, hidden_size)
        c_n-shape (D * num_layers, batch_size, hidden_size) """
        dec_output, (_, _) = self.lstm_decoder(enc_output, (h_0, c_0))

        # print(f"\ndec-output-shape: \n{dec_output.shape}")
        # print(f"\ndec-output: \n{dec_output}")

        return enc_output, dec_output

""" Runs the simulation. 
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: \n{device}")

    batch_size = 1 # 2
    seq_len = 5
    enc_input_size = 18
    dec_input_size = 5
    enc_hidden_size = 5
    dec_hidden_size = 18
    num_layers = 1
    x = torch.rand((batch_size, seq_len, num_features))

    # model definition
    model = EncoderDecoderLSTM(enc_input_size=enc_input_size, 
                               dec_input_size= dec_input_size,
                               enc_hidden_size=enc_hidden_size, 
                               dec_hidden_size=dec_hidden_size, 
                               num_layers=num_layers, 
                               device=device)        
    # model output
    enc_output, dec_output = model(x)

    


    # # other dimensions
    # batch_size = 2  # 32
    # seq_len = 5
    # enc_input_size = 18
    # dec_input_size = 18 #5
    # enc_hidden_size = 18
    # dec_hidden_size = 18
    # num_layers = 1
"""
