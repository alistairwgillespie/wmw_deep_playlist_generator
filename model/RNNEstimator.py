# torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn


class RNNEstimator(nn.Module):
    """
    RNN Estimator for generating sequences of target variables.
    """
    
    def __init__(self, input_features=9, hidden_dim=12, n_layers=2, output_dim=9, batch_size=12):
        super(RNNEstimator, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers
        self.batch_size = batch_size
        
        # RNN Layer
        self.rnn = nn.RNN(input_features, hidden_dim, n_layers, dropout=0.3)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def init_hidden(self):
        """
        Initialize the hidden and cell states of the LSTM with zeros.
        """
        return torch.zeros(self.hidden_layers, self.batch_size, self.hidden_dim)
        
    def forward(self, input, hidden_state):
        """
        Perform a forward pass of our model on batch of tracks.
        """
        
        # Passing in the input and hidden state into the model and obtaining outputs
        output, hidden_state = self.rnn(input.view(len(input), self.batch_size, -1), hidden_state)
        
        output = self.fc(output)
        
        return output, hidden_state
