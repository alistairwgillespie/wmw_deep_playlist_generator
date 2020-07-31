<p align="center"><img align="center" src="img/wilsons_fm.jpg" data-canonical-src="img/wilsons_fm.jpg" width="300" height="300" /></p>



# Wilson's Deep Playlist Generator

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 

## Background

Wilson's Deep Playlist Generator provides a pipeline for generating beautiful sequences of tracks. The generator uses a Recurrent Neural Network to learn the patterns used by playlist curators and artists to build playlists. The default model is trained on all volumes from the Wilson's Morning Wake Up series on Spotify.

To listen to the fruits of this project's labour, head on over to the playlist at this [link](https://open.spotify.com/playlist/7x1MY3AW3YCaHoicpiacGv?si=z5uRN003SN2fd1C7lyXBqw)

If you would like to listen to the original playlist, you can find it [here](https://open.spotify.com/playlist/2cczJrvEvS5j5oO5tf7ooP?si=lpxcB8a6TZqV6f_GLto8gw) too.

## Setup

1. Create virtual environment

```bash
conda env create --file local_env.yml
```

2. Activate and deactivate environment

```bash
conda activate local_wmw
conda deactivate
```

3. Add environment/kernel to Jupyter Notebook if you'd like to explore a little.

```bash
python -m ipykernel install --user --name=local_wmw
```

4. Install latest PyTorch for local and cpu only

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

5. Head on over to [Spotify for Developers](https://developer.spotify.com/dashboard/) and setup an app. Spotify will then supply an ID and Secret for accessing your app. Store these in your own .env file in the root of the directory like so:

```python
# example .env file
SPOTIFY_EMAIL=hello@example.com
SPOTIPY_CLIENT_ID=XXX
SPOTIPY_CLIENT_SECRET=XXX
SPOTIPY_REDIRECT_URI=http://localhost:8080
CACHE=.spotipyoauthcache

# Modeling info
MODEL_DIR=models
MODEL_CHOICE=lstm_model.pth
SCALER_CHOICE=standard_features.pkl
PCA_CHOICE=dim_red.pkl
```

## Run

To generate a playlist, run the following commands in the the command line whilst in the root of the repo:

```
conda activate local_wmw

# First generate a playlist pool
python get_playlist_pool.py

# Then generate a playlist
python main.py
```

A single playlist takes around 10 minutes to generate.


