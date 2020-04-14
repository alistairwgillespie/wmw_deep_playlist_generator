# Utilities
import os
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import joblib
from scipy.spatial.distance import cdist

# PyTorch
import torch
import torch.optim as optim

# Models
from model.LSTM_Estimator import LSTMEstimator
from model.RNN_Estimator import RNNEstimator

# Spotify API
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify variables
username = os.environ['SPOTIFY_EMAIL']
spotify_id = os.environ['SPOTIFY_ID']
spotify_secret = os.environ['SPOTIFY_SECRET']

# Set API scope
scope='playlist-read-private, playlist-modify-private, playlist-modify-public'

# Get auth token
token = util.prompt_for_user_token(username, 
                                   scope,
                                   client_id=spotify_id,
                                   client_secret=spotify_secret,
                                   redirect_uri='http://localhost/')

#Authenticate
sp = spotipy.Spotify(
    client_credentials_manager = SpotifyClientCredentials(
        client_id=spotify_id,
        client_secret=spotify_secret
    )
)


class Playlist():
    def __init__(self, wmw_pool, model_type="LSTM"):
        self.recommended_track_ids = pd.DataFrame() #list of track ids straight from spotify
        self.trax = [] #all tracks as dict
        self.df = None #this is where the data goes
        self.playlist = None
        self.wmw_pool_df = wmw_pool
        
        # Feature set
        self.feature_list =  ['danceability','energy', 'loudness', 'speechiness', 'acousticness',
                         'instrumentalness', 'liveness', 'valence', 'tempo']
    
        # Setup feature standardisation
        self.std_scaler = joblib.load('artefacts/standard_features.pkl')
        
        # Setup dimensionality reduction for track picking
        self.dim_red = joblib.load('artefacts/dim_red.pkl')
        
        if model_type == "LSTM":
            model = LSTMEstimator(9, 30, 1, 9)
            model.load_state_dict(torch.load('artefacts/lstm_model.pth'))
            
        elif model_type == "RNN":
            model = RNNEstimator(9, 30, 9)
            model.load_state_dict(torch.load('artefacts/rnn_model.pth'))
        else:
            print("Please specify either the RNN or LSTM model using the model_type parameter.")
        
        model.eval()
        
        # Start building the new playlist
        self.intro_track = self.get_first_track()
        self.new_playlist = self.predict_playlist(model, self.intro_track)

    
    def get_first_track(self):
        """Get first track based on recommendations."""
        # Sample an intro song from the WMW history
        song = self.wmw_pool_df[self.wmw_pool_df['position'] == 1].sample(1).copy()

        # Gather a recommendation based on the intro track using spotify
        song_res = sp.recommendations(seed_tracks = song['id'].values, limit=1)
        
        # Gather track freatures from spotify result
        for r in song_res['tracks']:
            track={}
            track['id'] = r['id']
            track['artists'] = [i['name'] for i in r['artists']],
            track['name'] = r['name']
            track_features = sp.audio_features(r['id'])[0]
            track.update(track_features)
            self.intro_track = pd.DataFrame(track, index=[0])

        # Prepare features
        self.intro_track[self.feature_list] = self.std_scaler.transform(self.intro_track[self.feature_list])
        
        return self.intro_track
    
    def harmonic_match(self, key, mode):
        """Given a key and mode, return compatible keys according to the harmonic wheel."""
        
        # Harmonic Mixing Wheel: Pitch Class 
        # 1A 0 - A flat minor: 8 | 1B 0 - B major: 11
        # 2A 1 - E flat minor: 3 | 2B 1 - F-sharp major: 6
        # 3A 2 - B-flat minor: 10 | 3B 2 - D-flat major: 1
        # 4A 3 - F minor: 5 | 4B 3 - A-flat major: 8
        # 5A 4 - C minor: 0 | 5B 4 - E-flat major: 3
        # 6A 5 - G minor: 7 | 6B 5 - B-flat major: 10
        # 7A 6 - D minor: 2 | 7B 6 - F major: 5
        # 8A 7 - A minor: 9 | 8B 7 - C major: 0
        # 9A 8 - E minor: 4 | 9B 8 - G major: 7
        # 10A 9 - B minor: 11 | 10B 9 - D major: 2
        # 11A 10 - F sharp minor: 6 | 11B 10 - A major: 9
        # 12A 11 - D flat minor: 1 | 12B 11 - E major: 4

        # Harmonic keys mapped to corresponding pitch classes
        pitch_to_harmonic_keys = {0: [4, 7], 1: [11, 2], 2: [6, 9],
                                  3: [1, 4], 4: [8, 11], 5: [3, 6],
                                  6: [10, 1], 7: [5, 8], 8: [0, 3],
                                  9: [7, 10], 10: [2, 5], 11: [9, 0]}

        # Extract values and keys
        dv = np.array(list(pitch_to_harmonic_keys.values()))
        dk = np.array(list(pitch_to_harmonic_keys.keys()))

        # Harmonic key code corresponding song pitch class
        harm_key = dv[np.where(dk == key)][0][mode]

        # Harmonic key codes
        harmonic_keys = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        # Get compatible key codes
        comp_keycodes = np.take(harmonic_keys, 
                                [harm_key - 1, harm_key, harm_key + 1],
                                mode='wrap')

        # Compatible keys
        comp_keys = [np.where(dv[:, mode] == i)[0][0].tolist() for i in comp_keycodes]

        # Compatible up/down key
        inner_outer_key = np.array([np.where(dv[:, int(not bool(mode))] == harm_key)[0][0]])

        comp_keys = np.concatenate([comp_keys, inner_outer_key])
        
        return comp_keys, inner_outer_key
    
    
    def get_position_recommendations(self, track_position):
        """Obtain a dataframe of recommended tracks for a specific track position."""
        
        recommendations = pd.DataFrame()

        wmw_sample = random.sample(self.wmw_pool_df['volume'].unique().tolist(), 10)

        wmw_sample_df = self.wmw_pool_df[
            (self.wmw_pool_df['volume'].isin(wmw_sample)) & 
            (self.wmw_pool_df['position'] == track_position)
        ]

#         wmw_sample_df = wmw_sample_df[].copy()

        # Iterate full catalog of WMW songs
        for _, row in wmw_sample_df.iterrows():
            
            song_search = row['track_name'].partition('-')[0] + ' ' + row['artist_name']
            
            try:

                # Query Spotify to get track metadata
                song_res = sp.search(song_search, limit=1)['tracks']['items'][0]

                # Gather recommendations for each of the past WMW tracks
                results = sp.recommendations(seed_tracks = [song_res['id']], limit=20)

                for r in results['tracks']:
                    track={}
                    track['id'] = r['id']
                    track['artists'] = [i['name'] for i in r['artists']],
                    track['name'] = r['name']
                    track_features = sp.audio_features(r['id'])[0]
                    track.update(track_features)
                    final_track = pd.DataFrame(track, index=[0])
                    recommendations = recommendations.append(final_track, ignore_index=True)

            except:
                print("Song not searchable")

        recommendations[self.feature_list] = self.std_scaler.transform(recommendations[self.feature_list])

        return recommendations
    
    def pick_optimal_track(self, candidates, target):
        """Select the track with the minimum distance between the candidate tracks."""
        
        candidates_reduced = self.dim_red.transform(candidates[self.feature_list])
        
        target_reduced = self.dim_red.transform(target)
        
        next_track_id = np.argmin(cdist(target_reduced, candidates_reduced))
        
        next_track = candidates.iloc[next_track_id]
        
        return next_track
    

    def predict_playlist(self, model, intro_track, playlist_len=15):
        """Predict playlist"""
        
        # Prepare prediction list
        predicted = intro_track
        
        # Prepare initial input 
        inp = torch.FloatTensor(intro_track[self.feature_list].values)
        
        print("Intro track:", predicted['name'].values[0], '-', ', '.join(predicted['artists'].values[0]))

        for p in tqdm(range(2, playlist_len + 1)):
            print("Track #%s - Generating candidates" % p)
            
            # Important stuff about the last track
            current_track = predicted.iloc[-1]
            current_key = current_track['key']
            current_mode = current_track['mode']

            # Generate output feature set of next song
            output = model(inp).detach().numpy()

            # Get mode and key from last song and generate compatible keys and modes
            keys, outer_inner_key = self.harmonic_match(current_key, current_mode)

            # Get recommended tracks for current track position
            recommendations = self.get_position_recommendations(p)
            
            print("Recommendations", recommendations.shape)
            
            # Filter for compatible tracks according to key and mode (harmonic wheel)
            next_tracks_curr_mode = recommendations[
                (recommendations['key'].isin(keys[:3])) & (recommendations['mode'] == current_mode)
            ]
            
            print("Curr mode", next_tracks_curr_mode.shape)
            
            next_tracks_change_mode = recommendations[
                (recommendations['key'] == keys[-1]) & (recommendations['mode'] == abs(int(not current_mode)))
            ]
            
            print("Change mode", next_tracks_change_mode.shape)
            
            candidate_tracks = pd.concat([next_tracks_curr_mode, next_tracks_change_mode]).reset_index(drop=True)
            
            # Ensure no duplicates exist in the playlist
            candidate_tracks = candidate_tracks[~candidate_tracks['id'].isin(predicted['id'])]
            
            print("CANDIDATES:", candidate_tracks.shape)
            
            # Pick optimal track
            next_track = self.pick_optimal_track(candidate_tracks, output)
            
            print("Selected:", next_track['name'], '-', ', '.join(next_track['artists']))

            # Set new input vector for next song
            inp = torch.FloatTensor([next_track[self.feature_list]])

            # Append next song to playlist
            predicted = predicted.append(next_track, ignore_index=True)
            
            print('-' * 20)

        return predicted
    

    def post_playlist(self):
        if token:
            sp = spotipy.Spotify(auth=token)
            sp.trace = False
            tracks = sp.user_playlist_replace_tracks('1247785541', '7x1MY3AW3YCaHoicpiacGv', self.new_playlist['id'].values)
            print("Posting latest Wilson's FM.")
        else:
            print("Can't get token for", username)
