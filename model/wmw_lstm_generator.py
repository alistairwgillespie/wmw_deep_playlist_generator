# torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn


class WMWGenerator(nn.Module):
    """
    LSTM Estimator for generating sequences of target variables.
    """

    def __init__(self, input_features=9, hidden_dim=12, n_layers=2, output_dim=9, batch_size=12):
        """
        Initialize the LSTM model.
        """
        super(WMWGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers
        self.batch_size = batch_size

        # The LSTM takes track features as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(input_features, self.hidden_dim, n_layers, dropout=0.3)

        # Connected layer to output
        self.hidden2target = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        """
        Initialize the hidden and cell states of the LSTM with zeros.
        """
        return (torch.zeros(self.hidden_layers, self.batch_size, self.hidden_dim)), \
               (torch.zeros(self.hidden_layers, self.batch_size, self.hidden_dim))

    def forward(self, input, hidden_cell):
        """
        Perform a forward pass of our model on batch of tracks.
        """
        
        # define the feedforward behavior
        lstm_out, hidden_cell = self.lstm(input.view(len(input), self.batch_size, -1), hidden_cell)

        output = self.hidden2target(lstm_out)
        
        return output, hidden_cell
