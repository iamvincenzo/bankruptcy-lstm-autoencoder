import torch.nn as nn


""" Class for LSTM AutoEncoder Model. """
class LSTMAutoencoder(nn.Module):
    """ Initialize configuration. """
    def __init__(self, seq_len=3, n_features=2, embedding_dim=128, input_dim=128):
        super(LSTMAutoencoder, self).__init__()
        # encoder
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        # decoder
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.encoder_lstm1 = nn.LSTM(input_size=n_features,
                                     hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=self.hidden_dim,
                                     hidden_size=embedding_dim, num_layers=1, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(input_size=input_dim,
                                     hidden_size=input_dim, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=input_dim, 
                                    hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    """ Method used to define the computations needed to create the 
        output of the neural network given its input. """
    def forward(self, x):
        # encoder
        x = x.reshape((1, self.seq_len, self.n_features))
        print(f"x-reshape: {x.shape}")
        x, (_, _) = self.encoder_lstm1(x)
        print(f"x-enclstm1: {x.shape}")
        x, (hidden_n, _) = self.encoder_lstm2(x)
        print(f"x-enclstm2: {x.shape}")
        self.n_features = 1
        x = hidden_n.reshape((self.n_features, self.embedding_dim))  
        print(f"x-reshape: {x.shape}")

        # decoder
        x = x.repeat(self.seq_len, self.n_features)
        print(f"x-repeat: {x.shape}")
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        print(f"x-reshape: {x.shape}")
        x, (hidden_n, cell_n) = self.decoder_lstm1(x)
        print(f"x-declstm1: {x.shape}")
        x, (hidden_n, cell_n) = self.decoder_lstm2(x)
        print(f"x-declstm2: {x.shape}")
        x = x.reshape((self.seq_len, self.hidden_dim))
        print(f"x-reshape: {x.shape}")
        x = self.output_layer(x)

        return x


""" Test: 
if __name__ == "__main__":
    import torch
    x = torch.rand(1, 5, 18)
    print(f"x-input: \n{x}\n")
    print(f"\nx-shape: {x.shape}")

    model = LSTMAutoencoder(seq_len=5, n_features=18, 
                            embedding_dim=128, input_dim=128)
    output = model(x)    
    print(f"out-shape: {output.shape}\n")
    print(f"x-output: \n{output}\n")
"""
