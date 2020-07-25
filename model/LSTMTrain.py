import subprocess as sb 
import sys 

sb.call([sys.executable, "-m", "pip", "install", 'pandas']) 

import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from LSTMEstimator import LSTMEstimator
from PlaylistDataset import PlaylistDataset


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'lstm_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEstimator(model_info['input_features'], model_info['hidden_dim'], model_info['hidden_layers'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'lstm.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    
    dataset =  PlaylistDataset(training_dir, "tensor_train.csv")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script of the LSTM model. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    model.train() # Make sure that the model is in training mode.

    for epoch in range(1, epochs + 1):

        avg_loss = 0

        # Iterate over dataset
        for i, batch in enumerate(train_loader):
            # Clear stored gradient
            optimizer.zero_grad()

            # Initialize hidden state
            hidden_cell = model.init_hidden()

            # Batch of 12 tracks
            batch_x = batch[0]  # X input
            batch_y = batch[-1]  # y target

            # Forward pass
            output, hidden_cell = model(batch_x.unsqueeze(0), hidden_cell)

            # Calculate MAE loss over batch
            batch_loss = criterion(output.squeeze(0), batch_y)
            avg_loss += batch_loss.item()

            # Zero out gradient, so it doesnt accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            batch_loss.backward()

            # Update parameters
            optimizer.step()

        if epoch % 50 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(avg_loss / len(train_loader)))

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Model Parameters    
    parser.add_argument('--input_features', type=int, default=9, metavar='N',
                        help='size of the feature space (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=30, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--hidden_layers', type=int, default=1, metavar='N',
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--output_dim', type=int, default=9, metavar='N',
                        help='size of the output dimension (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
      
    ## Build the model by passing in the input params
    model = LSTMEstimator(args.input_features, args.hidden_dim, args.hidden_layers, args.output_dim).to(device)

    ## Define an optimizer andfunction for training
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.L1Loss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)
    
    model_info_path = os.path.join(args.model_dir, 'lstm_info.pth')
    
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)
        
    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'lstm.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)