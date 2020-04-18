# torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn


class RNNEstimator(nn.Module):
    """
    RNN Estimator for generating sequences of target variables.
    """
    
    def __init__(self, input_features=9, hidden_dim=30, n_layers=1, output_dim=9):
        super(RNNEstimator, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_features, hidden_dim, n_layers, dropout=0.3)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    ## Initialize the hidden and cell states of the LSTM with zeros.
    def init_hidden(self):
        return torch.zeros(self.hidden_layers, 1, self.hidden_dim)
        
    ## Define the feedforward behavior of the network
    def forward(self, input, hidden_state):
        
        # Passing in the input and hidden state into the model and obtaining outputs
        output, hidden_state = self.rnn(input.view(len(input) ,1, -1), hidden_state)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        output = self.fc(output.contiguous().view(len(input), -1))
        
        return output, hidden_state
