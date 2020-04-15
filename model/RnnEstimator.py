# torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn


class RnnEstimator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RnnEstimator, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
    
    ## Define the feedforward behavior of the network
    def forward(self, input_sequence, hidden):
        combined = torch.cat((input_sequence, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)