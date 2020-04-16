import subprocess as sb
import sys
import os
# import random
# import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
# from scipy.spatial.distance import cdist
import config
from playlist import Playlist

# PyTorch
# import torch
# import torch.optim as optim

# Models
# from model.LstmEstimator import LstmEstimator
# from model.RnnEstimator import RnnEstimator

# If AWS SageMaker:

# Joblib
# sb.call([sys.executable, "-m", "pip", "install", 'joblib'])
# import joblib

# Spotify API
sb.call([sys.executable, "-m", "pip", "install", 'spotipy'])
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

def main():
    # Spotify variables
    username = config.SPOTIFY_EMAIL
    spotify_id = config.SPOTIFY_ID
    spotify_secret = config.SPOTIFY_SECRET

    # Set API scope
    scope = "playlist-read-private, playlist-modify-private, playlist-modify-public"

    # Get auth token
    token = util.prompt_for_user_token(username,
                                       scope,
                                       client_id=spotify_id,
                                       client_secret=spotify_secret,
                                       redirect_uri='http://localhost/'
                                       )

    # Authenticate
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=spotify_id,
            client_secret=spotify_secret
        )
    )
    data_dir = 'data'

    track_data = pd.read_csv(os.path.join(data_dir, "wmw_tracks.csv"))

    playlist = Playlist(track_data, sp, token, model_type="LSTM")

    playlist.post_playlist()

if __name__ == '__main__':
    main()