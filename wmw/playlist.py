class Playlist():
    def setup_model(self, dir, filename):
        """Setup model in eval mode."""
        pass

    def get_first_track(self):
        """Get first track based on recommendations."""
        pass

    def harmonic_match(self, key, mode):
        """Given a key and mode, return compatible keys according to the harmonic wheel."""
        pass

    def get_recommendations(self):
        """Obtain a dataframe of recommended tracks for a specific track position."""
        pass

    def pick_optimal_track(self, candidates, target):
        """Select the track with the minimum Euclidean distance between the candidate tracks."""
        pass

    def generate_playlist(self, model, intro_track, playlist_len=15):
        """Generate playlist."""
        pass

    def post_playlist(self):
        """Posts playlist.

        Raises:
            SpotifyException: When query fails to gather playlist metadata.
        """
        pass
