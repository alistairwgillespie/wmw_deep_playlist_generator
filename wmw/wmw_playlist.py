import subprocess as sb
import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
from wmw.playlist import Playlist
from wmw.lstm_generator import LSTMGenerator
from dotenv import load_dotenv, find_dotenv

# Spotify API
sb.call([sys.executable, "-m", "pip", "install", 'spotipy'])
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.client import SpotifyException

# Joblib
sb.call([sys.executable, "-m", "pip", "install", 'joblib'])
import joblib

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# ENV variables
MODEL_DIR = os.environ.get("MODEL_DIR")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE")
SCALER_CHOICE = os.environ.get("SCALER_CHOICE")
PCA_CHOICE = os.environ.get("PCA_CHOICE")


# TODO: Contain recommendations pool to Best Of and add 
# Deep Work as pool for intro tracks
class WMWPlaylist(Playlist):
    def __init__(self, track_pool, spotify_auth, username, playlist_id, model_type="LSTM"):
        """ Initiates pool of historic tracks, spotify api authentication and
            model of choice.
        """
        self.recommended_track_ids = pd.DataFrame()
        self.playlist_id = playlist_id
        self.track_pool = track_pool
        self.spotify_auth = spotify_auth
        self.username = username

        # Feature set
        self.feature_list = ['danceability', 'energy',
                             'loudness', 'speechiness',
                             'acousticness', 'instrumentalness',
                             'liveness', 'valence', 'tempo']

        # Setup feature and model pipeline
        self.std_scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_CHOICE))
        self.dim_red = joblib.load(os.path.join(MODEL_DIR, PCA_CHOICE))
        self.model = self.setup_model(MODEL_DIR, MODEL_CHOICE)

        # Initiate pipeline
        self.intro_track = self.get_first_track()
        self.new_playlist = self.generate_playlist(self.model, self.intro_track)

    def setup_model(self, dir, filename):
        """Setup model in eval mode."""
        try:
            model = LSTMGenerator(9, 12, 2, 9, 1)
            model.load_state_dict(torch.load(os.path.join(dir, filename)))
            model.eval()
        except Exception as exc:
            print(f'{exc} Failed to import model.')
        return model

    def get_first_track(self):
        """Get first track based on recommendations."""
        # Sample an intro song from track pool
        song = self.track_pool.sample(1).copy()

        # Gather a recommendation based on the intro track using spotify
        song_res = self.spotify_auth.recommendations(seed_tracks=song['id'].values, limit=1)

        # Gather track freatures from spotify result
        for r in song_res['tracks']:
            track = {'id': r['id'], 'artists': ([i['name'] for i in r['artists']],), 'name': r['name']}
            track_features = self.spotify_auth.audio_features(r['id'])[0]
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

    def get_recommendations(self, sample_size=15, recommend_size=15):
        """Obtain a dataframe of recommended tracks for a specific track position."""
        recommendations = pd.DataFrame()
        sample_df = self.track_pool.sample(sample_size).copy()

        # Iterate full catalog of WMW songs
        for _, row in sample_df.iterrows():
            song_search = row['track_name'].partition('-')[0] + ' ' + row['artist_name']
            try:
                song_res = self.spotify_auth.search(song_search, limit=1)['tracks']['items'][0]
                results = self.spotify_auth.recommendations(seed_tracks=[song_res['id']], limit=10)
                for r in results['tracks']:
                    track = {'id': r['id'], 'artists': [i['name'] for i in r['artists']], 'name': r['name']}
                    track_features = self.spotify_auth.audio_features(r['id'])[0]
                    track.update(track_features)
                    final_track = pd.DataFrame(track)
                    recommendations = recommendations.append(final_track, ignore_index=True)
            except Exception as exc:
                print(f'{exc}: sample song not searchable.')
        recommendations[self.feature_list] = self.std_scaler.transform(recommendations[self.feature_list])
        return recommendations

    def pick_optimal_track(self, candidates, target):
        """Select the track with the minimum Euclidean distance between the candidate tracks."""
        candidates_reduced = self.dim_red.transform(candidates[self.feature_list])
        target_reduced = self.dim_red.transform(target[0])
        next_track_id = np.argmin(cdist(target_reduced, candidates_reduced))
        next_track = candidates.iloc[next_track_id]
        return next_track


    def generate_playlist(self, model, intro_track, playlist_len=15):
        """Generate playlist."""

        # Prepare prediction list
        predicted = intro_track

        # Prepare initial input
        input = torch.FloatTensor(intro_track[self.feature_list].values)
        
        print("Track #1")
        print("Selected:", predicted['name'].values[0], '-', ', '.join(predicted['artists'].values[0]))

        hidden_state = model.init_hidden()

        for p in range(2, playlist_len + 1):
            print("Track #%s" % p)

            # Important stuff about the last track
            current_track = predicted.iloc[-1]
            current_key = current_track['key']
            current_mode = current_track['mode']

            # Generate output feature set of next song
            output, hidden_state = model(input, hidden_state)

            output = output.detach().numpy()

            # Get mode and key from last song and generate compatible keys and modes
            keys, outer_inner_key = self.harmonic_match(current_key, current_mode)

            # Get recommended tracks for current track position
            recommendations = self.get_recommendations()

            # Filter for compatible tracks according to key and mode (harmonic wheel)
            next_tracks_curr_mode = recommendations[
                (recommendations['key'].isin(keys[:3])) & (recommendations['mode'] == current_mode)
            ]

            next_tracks_change_mode = recommendations[
                (recommendations['key'] == keys[-1]) & (recommendations['mode'] == abs(int(not current_mode)))
            ]

            candidate_tracks = pd.concat([next_tracks_curr_mode, next_tracks_change_mode]).reset_index(drop=True)

            # Ensure no duplicates exist in the playlist
            candidate_tracks = candidate_tracks[~candidate_tracks['id'].isin(predicted['id'])]

            # Pick optimal track
            next_track = self.pick_optimal_track(candidate_tracks, output)

            print("Selected:", next_track['name'], '-', next_track['artists'])

            # Set new input vector for next song
            input = torch.FloatTensor([next_track[self.feature_list]])

            # Append next song to playlist
            predicted = predicted.append(next_track, ignore_index=True)

            print('-' * 20)

        return predicted


    def post_playlist(self):
        """Posts playlist.

        Raises: 
            SpotifyException: When query fails to gather playlist metadata.
        """
        try:
            self.spotify_auth.user_playlist_replace_tracks(
                self.username, 
                self.playlist_id,
                self.new_playlist['id'].values)
        except SpotifyException as e:
            print(f"{e}: Failed to post playlist to Spotify.")