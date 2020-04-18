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

from RNNEstimator import RNNEstimator
from PlaylistDataset import PlaylistDataset


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}s
    model_info_path = os.path.join(model_dir, 'rnn_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNEstimator(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'rnn.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    
    dataset =  PlaylistDataset(training_dir, "tensor_train.csv")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # training loop is provided
    for epoch in range(1, epochs + 1):
        
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()

            cum_loss = 0
            
            hidden_cell = model.init_hidden()
            
            for i, track in enumerate(batch):
                
                track_x = track[0]
                track_y = track[-1]
                
                output, hidden_cell = model(track_x.unsqueeze(0), hidden_cell)
                
                loss = criterion(output.squeeze(0), track_y)
                loss.backward(retain_graph=True)
                optimizer.step()
                cum_loss += loss.data.item()

            total_loss = cum_loss / len(batch[0])
            
        if epoch % 50 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(total_loss))


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
                        help='input batch size for training (default: 12)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Model Parameters    
    parser.add_argument('--input_features', type=int, default=9, metavar='N',
                        help='size of the feature space (default: 9)')
    parser.add_argument('--hidden_dim', type=int, default=30, metavar='N',
                        help='size of the hidden dimension (default: 30)')
    parser.add_argument('--output_dim', type=int, default=9, metavar='N',
                        help='size of the output dimension (default: 9)')
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
    model = RNNEstimator(args.input_features, args.hidden_dim, args.output_dim).to(device)

    ## Define an optimizer andfunction for training
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.L1Loss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)
    
    model_info_path = os.path.join(args.model_dir, 'rnn_info.pth')
    
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)
    

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'rnn.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)