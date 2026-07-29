from pathlib import Path

from src.lana_nlp.scripts.data_loader import LyricsDataLoader
from src.lana_nlp.scripts.lyrics_analyzer import LyricsAnalyzer
from src.lana_nlp.scripts.text_cleaner import TextCleaner

# Load the lyrics csv
loader = LyricsDataLoader(
    Path("../data/raw/lyrics.csv")
)

df = loader.load()

print(df["lyrics"].isna().sum())

# Clean text
cleaner = TextCleaner()

df["basic_lyrics"] = df["lyrics"].apply(
    cleaner.basic_clean
)

df["nlp_tokens"] = df["lyrics"].apply(
    cleaner.nlp_clean
)

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