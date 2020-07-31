import pandas as pd
import spotipy
import os
from dotenv import load_dotenv, find_dotenv
from spotipy.client import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pickle import dump

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

# Exclude list
exclude_list = [
    'track_name', 'artist_name', 'duration_ms', 
    'track_href', 'uri', 'time_signature', 'id', 
    'type', 'analysis_url', 'mode','key']


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


def generate_dataset(sp, playlists, search, save_as):
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
                if "." in search:
                    current_volume = playlist['name'].split('.')[1]
                else:
                    current_volume = 1
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
        tracks_df.to_csv("data/" + save_as)
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

# # Search playlists and get track data
# input_search = 'Morning Wake Up Vol.'
# sp = connect_spotify()
# playlists = get_playlist_metadata(sp, n=50)

# # Generate WMW dataset and save
# tracks_df = generate_dataset(sp, playlists, input_search, 'wmw_dataset.csv')

# # Build training dataset
# tracks_df, _ = standardize(tracks_df, feature_list, exclude_list)
# training_df = pd.DataFrame()

# for i in tracks_df['volume'].unique():
#     X_df = tracks_df[tracks_df['volume'] == i].iloc[:13]
#     y_df = tracks_df[tracks_df['volume'] == i].iloc[:13].shift(-1)[feature_list]
#     X_y_df = pd.concat([X_df, y_df.add_prefix('y_')], axis=1).dropna()
#     training_df = training_df.append(X_y_df, ignore_index=True)
    
#     del X_df
#     del y_df
#     del X_y_df

# training_df.to_csv("data/tensor_train.csv", index=False)

# Connect to Spotify
sp = connect_spotify()

# # Generate WMW Recommendation pool
# input_search = "Wilson's Morning Wake Up Best Of"
# playlists = get_playlist_metadata(sp, n=50)
# generate_dataset(sp, playlists, input_search, 'wmw_pool.csv')

# Generate LoFi Recommendation pool
input_search = "Lo-Fi Butter | Beats to study/relax to"
playlists = get_playlist_metadata(sp, n=50)
generate_dataset(sp, playlists, input_search, 'lofi_pool.csv')

# # Generate Classical Recommendation pool
# input_search = "Wilson's Deep Work II"
# playlists = get_playlist_metadata(sp, n=50)
# generate_dataset(sp, playlists, input_search, 'classical_pool.csv')