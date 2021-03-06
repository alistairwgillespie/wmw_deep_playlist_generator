import subprocess as sb
import sys
import os
import pandas as pd
from wmw.wmw_playlist import WMWPlaylist
import os
from dotenv import load_dotenv, find_dotenv

# Spotify API
sb.call([sys.executable, "-m", "pip", "install", 'spotipy'])
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.client import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

# Spotify variables
USERNAME = os.environ.get("SPOTIFY_EMAIL")
SPOTIFY_ID = os.environ.get("SPOTIPY_CLIENT_ID")
SPOTIFY_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
CACHE = os.environ.get("CACHE")


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


def main():

    # username = config.SPOTIFY_EMAIL
    # spotify_id = config.SPOTIFY_ID
    # spotify_secret = config.SPOTIFY_SECRET

    # # Set API scope
    # scope = "playlist-read-private, playlist-modify-private, playlist-modify-public"

    # # Get auth token
    # token = util.prompt_for_user_token(username,
    #                                    scope,
    #                                    client_id=spotify_id,
    #                                    client_secret=spotify_secret,
    #                                    redirect_uri='http://localhost:8080'
    #                                    )

    # # Authenticate
    # sp = spotipy.Spotify(
    #     client_credentials_manager=SpotifyClientCredentials(
    #         client_id=spotify_id,
    #         client_secret=spotify_secret
    #     )
    # )
    data_dir = 'data'
    sp = connect_spotify()

    # # Wilson's FM
    # wmw_data = pd.read_csv(os.path.join(data_dir, "wmw_pool.csv"))
    # wmw_playlist = Playlist(wmw_data, sp, USERNAME, "5Yia6eXihjr96S3oyRm1dX")
    # wmw_playlist.post_playlist()

    # Wilson's Daily Lo-Fi
    lofi_data = pd.read_csv(os.path.join(data_dir, "lofi_pool.csv"))
    lofi_playlist = WMWPlaylist(lofi_data, sp, USERNAME, "5OwXRuFy7UJdyAfUkv59Ur")
    lofi_playlist.post_playlist()

    # # Wilson's Daily Classical
    # classical_data = pd.read_csv(os.path.join(data_dir, "classical_pool.csv"))
    # classical_playlist = Playlist(classical_data, sp, USERNAME, "4jFQH5eT0Q6XD95SpPyFcH")
    # classical_playlist.post_playlist()

if __name__ == '__main__':
    main()