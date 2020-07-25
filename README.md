<p align="center"><img align="center" src="img/wilsons_fm.jpg" data-canonical-src="img/wilsons_fm.jpg" width="300" height="300" /></p>



# Wilson's FM - A Deep Playlist Generator

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 

## Background

Wilson's FM is a project born out of a passion for curating playlists that move people as they go about their day; whether it be getting ready for work, setting foot in the gym or going for a walk, music can make things extra special.  

For the past two years, I have been curating a monthly playlist - aptly named Wilson's Morning Wake Up - that is designed to gently build as you start your day. The idea is to wake up, have a coffee then press play. Since the inaugural release, it now equates to 39 volumes (at the time of writing) and has over 200 followers.

This project is an attempt to train a sequence-based model that learns how I curate the Wilson's Morning Wake Up playlists. The intent is to leverage this to generate unique, blissful compilations of tunes each day via the Wilson's FM API.

In the future, I hope to generalize this framework into a tool that artists, playlist curators and alike, can use to generate quality playlists, more frequently, for their listeners.

To listen to the fruits of the this project's labour, head on over to the playlist at this [link](https://open.spotify.com/playlist/7x1MY3AW3YCaHoicpiacGv?si=z5uRN003SN2fd1C7lyXBqw)

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

5. Head on over to [link](https://developer.spotify.com/dashboard/) and setup an app. Spotify will then supply an ID and Secret for accessing your app. Store these in your own .env file in the root of the directory like so:

```python
# example .env file
SPOTIFY_EMAIL=example@example.com
SPOTIFY_ID=12345678910
SPOTIFY_SECRET=12345678910
```

## Run

To generate a playlist, run the following commands in the the command line whilst in the root of the repo:

```
conda activate local_wmw
python main.py
```

A single playlist takes around 10 minutes to generate.

## Structure

The project is structured like so:

```bash
.
|-- artefacts/ # Save models and artefacts here
	|-- dim_red.pkl # Principal Component Analysis
	|-- lstm_model.pth # Long Short-Term Memory Neural Network
	|-- rnn_model.pth # Vanilla Recurrent Neural Network
	|-- standard_features.pkl # Standard Scaler
|-- data/
    |-- tensor_train.csv # Training dataset
    |-- wmw.csv # Pool of Wilson's Morning Wake Up tracks to date
|-- model/
    |-- LSTMEstimator.py # LSTM Model with initialisation and feed-forward
    |-- LSTMTrain.py # Code for training the LSTM on AWS SageMaker
    |-- PlaylistDataset.py # Dataset Class
    |-- Predict.py # Code for predictions on AWS SageMaker
    |-- RNNEstimator.py # RNN Model with initialisation and feed-forward
    |-- RNNTrain.py # Code for training the RNN on AWS SageMaker
|-- img/
    |-- ...
|-- .gitignore # ...
|-- 0_Setup_Database.ipynb # Databased Setup for future use
|-- 1_Explore.ipynb # Initial data ingestion and analaysis
|-- 2_Feature_Engineering.ipynb # Feature preparation and further analysis
|-- 3_Train_Deploy_LOCAL.ipynb # Pipeline for training each model locally
|-- 4_Train_Deploy_AWS.ipynb # Pipeline for training each model on AWS SageMaker
|-- 5_Generate.ipynb # Generates a playlist and posts to Spotify
|-- LICENSE # MIT License
|-- local_env.yml # Environment details
|-- main.py # Pipeline that generates a playlist and posts to Spotify via CLI
|-- playlist.py # Playlist class
|-- PROPOSAL.md # Project Proposal
|-- README.md # ...
|-- REPORT.md # Project Report
```

