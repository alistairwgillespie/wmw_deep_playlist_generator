from wmw.playlist_dataset import PlaylistDataset
from wmw.lstm_generator import LSTMGenerator
from wmw.rnn_generator import RNNGenerator
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# should be the name of directory you created to save your features data
data_dir = 'data'
dataset =  PlaylistDataset(data_dir, "tensor_train.csv")
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)


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
    loss_record = []
    for epoch in range(1, epochs + 1):
        
        avg_loss = 0
        
        # Iterate over dataset
        for _, batch in enumerate(train_loader):
            
            # Clear stored gradient
            optimizer.zero_grad()
            
            # Initialize hidden state 
            hidden_cell = model.init_hidden()
            
            # Batch of 12 tracks
            batch_x = batch[0] # X input
            batch_y = batch[-1] # y target
            
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
        
        loss_record.append(avg_loss / len(train_loader))
        
        if epoch % 2 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(avg_loss / len(train_loader)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMGenerator(9, 12, 2, 9)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = torch.nn.L1Loss()
num_epochs = 100
train(lstm_model, dataloader, num_epochs, loss_fn, optimizer, device)
torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')