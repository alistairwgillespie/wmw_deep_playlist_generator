![](C:\Users\gilleal\Pictures\wilsons_fm (2).jpg )
# Wilson's FM - A Deep Playlist Generator

## Background

Wilson's FM is a project born out of a passion for curating playlists that move people as they go about their day; whether it be getting ready for work, setting foot in the gym or going for a walk, music can make things extra special.  

For the past two years, I have been curating a monthly playlist - aptly named Wilson's Morning Wake Up - that is designed to gently build as you start your day. The idea is to wake up, have a coffee then press play. Since the inaugural release, it now equates to 39 volumes (at the time of writing) and has over 200 followers.

This project is an attempt to train a sequent model that learns how I curate the Wilson's Morning Wake Up playlists. The intent is to leverage this to generate unique, blissful compilations of tunes each day via the Wilson's FM API.

In the future, I hope to generalize this framework into a tool that artists, playlist curators and alike, can use to generate quality playlists, more frequently, for their listeners.

To listen to the fruits of the this project's labour, head on over to the playlist at this [link](https://open.spotify.com/playlist/7x1MY3AW3YCaHoicpiacGv?si=z5uRN003SN2fd1C7lyXBqw)

## Setup

Create virtual environment

```bash
conda env create --file local_env.yml
```

Activate and deactivate environment

```bash
conda activate local_wmw
conda deactivate
```

Add environment/kernel to Jupyter Notebook

```bash
python -m ipykernel install --user --name=local_wmw
```

Install latest PyTorch for local and cpu only

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

## Proposal	