import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class PlaylistDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        """
        Args:
            data_dir (string): directory name
            csv_path (string): csv filename
        """
        # Read the csv file
        self.data = pd.read_csv(os.path.join(data_dir, csv_path))
        # First column contains the image paths
        self.data_arr = self.data.iloc[:, 2:11].values
        # Second column is the labels
        self.label_arr = self.data.iloc[:, 11:].values
        # Calculate len
        self.data_len = len(self.data.index)

    def __getitem__(self, index):
        # Get track
        single_track = torch.from_numpy(self.data_arr[index]).float()
        
        # Get label(class) of the image based on the cropped pandas column
        single_target = torch.from_numpy(self.label_arr[index]).float()

        return (single_track, single_target)

    def __len__(self):
        return self.data_len