import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spotipy
import spotipy.util as util
import os
import sys
import requests
from dotenv import load_dotenv, find_dotenv
from spotipy.client import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from pickle import dump
import joblib

# TODO: Build more user friendly structure for user input
# TODO: ASYNC for many playlists

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

# Spotify variables
USERNAME = os.environ.get("SPOTIFY_EMAIL")
SPOTIFY_ID = os.environ.get("SPOTIPY_CLIENT_ID")
SPOTIFY_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
CACHE = os.environ.get("CACHE")

# Feature list
feature_list = [
    'danceability','energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo']

exclude_list = [
    'track_name', 'artist_name', 'duration_ms', 
    'track_href', 'uri', 'time_signature', 'id', 
    'type', 'analysis_url', 'mode','key']

# # 
# AUTH_URL = 'https://accounts.spotify.com/api/token'

# # POST
# auth_response = requests.post(AUTH_URL, {
#     'grant_type': 'client_credentials',
#     'client_id': spotify_id,
#     'client_secret': spotify_secret,
# })

# # convert the response to JSON
# auth_response_data = auth_response.json()

# # save the access token
# access_token = auth_response_data['access_token']

# headers = {
#     'Authorization': 'Bearer {token}'.format(token=access_token)
# }

# print(headers['Authorization'])

# # Set API scope
# scope='playlist-read-private, playlist-modify-private, playlist-modify-public'

# # Get auth token
# token = util.prompt_for_user_token(username, 
#                                    scope,
#                                    client_id=spotify_id,
#                                    client_secret=spotify_secret,
#                                    redirect_uri='http://localhost/')

CACHE = '.spotipyoauthcache'

def connect_spotify():
    """Connects to Spotify API.

    Raises: 
        SpotifyException: When the inputs failt to authenticate with Spotify.
    """
    try:
        scope='playlist-read-private, playlist-modify-private, playlist-modify-public'
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, cache_path=CACHE))
        return sp
    except SpotifyException as e:
        print(f"{e}: Failed to connect to Spotify API.")


def get_playlist_metadata(sp, n):
    """Gets user's current playlist catalogue.

    Raises: 
        SpotifyException: When query fails to gather playlist metadata.
    """
    try:
        playlists = sp.current_user_playlists(limit=n)
        return playlists
    except SpotifyException as e:
        print(f"{e}: Failed to gather current user playlists.")


def generate_dataset(sp, playlists, search):
    """Gathers playlist(s) based on input search
    """
    tracks_df = pd.DataFrame()   
    print("Gathering playlist track data")
    print('-'*30)
    
    while playlists:
        for _, playlist in enumerate(playlists['items']):
            if search in playlist['name']:
                print("%s" % (playlist['name']))
                tracks = sp.playlist_tracks(playlist['uri'])
                current_volume = playlist['name'].split('.')[1]
                for j, track in enumerate(tracks['items']):
                    track_data={}
                    track_data['volume'] = current_volume
                    track_data['position'] = j + 1
                    track_data['track_name'] = track['track']['name']
                    track_data['artist_name'] = track['track']['artists'][0]['name']
                    track_features = sp.audio_features(track['track']['id'])[0]
                    track_data.update(track_features)
                    stage = pd.DataFrame(track_data, index=[0])
                    tracks_df = tracks_df.append(stage, ignore_index=True)
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
        tracks_df.to_csv("data/dataset.csv")
        return tracks_df


def standardize(df, feature_list, exclude_list):
    """Fit and save StandardScaler and PCA
    """

    # Exclude unnecessary columns
    df.drop(columns=exclude_list, inplace=True)
    # Fit and save standardizer
    standard_scaler = StandardScaler()
    # standard_scaler.fit(songs_df[feature_list])
    standard_features = standard_scaler.fit_transform(df[feature_list])
    processed_df = pd.DataFrame(standard_features, index=df.index, columns=df.columns[2:])
    tracks_df = pd.concat([df[['volume', 'position']], processed_df[feature_list]], axis=1)
    # Fit and save PCA
    pca = PCA(n_components=3)
    # pca.fit(df[feature_list])
    pca_df = pca.fit_transform(df[feature_list])

    # Save
    dump(standard_scaler, open('models/standard_features.pkl', 'wb'))
    dump(pca, open('models/dim_red.pkl', 'wb'))

    return df, pca_df

# Search playlists and get track data
input_search = 'Morning Wake Up Vol.'
sp = connect_spotify()
playlists = get_playlist_metadata(sp, n=50)
tracks_df = generate_dataset(sp, playlists, input_search)

# Build and save feature models
tracks_df, _ = standardize(tracks_df, feature_list, exclude_list)

# Prepare labels using shift
training_df = pd.DataFrame()

for i in tracks_df['volume'].unique():
    X_df = tracks_df[tracks_df['volume'] == i].iloc[:13]
    y_df = tracks_df[tracks_df['volume'] == i].iloc[:13].shift(-1)[feature_list]
    X_y_df = pd.concat([X_df, y_df.add_prefix('y_')], axis=1).dropna()
    training_df = training_df.append(X_y_df, ignore_index=True)
    
    del X_df
    del y_df
    del X_y_df

training_df.to_csv("data/tensor_train.csv", index=False)