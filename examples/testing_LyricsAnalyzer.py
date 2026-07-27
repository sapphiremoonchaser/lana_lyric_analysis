from pathlib import Path

from src.lana_nlp.scripts.data_loader import LyricsDataLoader
from src.lana_nlp.scripts.lyrics_analyzer import LyricsAnalyzer

# Load the lyrics csv
loader = LyricsDataLoader(
    Path("../data/raw/lyrics.csv")
)

df = loader.load()

# Create the analyzer to get stats
analyzer = LyricsAnalyzer(df)

# Total number of songs
number_of_songs = analyzer.number_of_songs()

# List of albums
albums = analyzer.albums()

# Number of songs by album
songs_by_album = analyzer.number_of_songs_by_album()

# Song length stats
song_length_stats = analyzer.song_length_stats()

x = 1