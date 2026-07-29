from pathlib import Path

from src.lana_nlp.scripts.data_loader import LyricsDataLoader
from src.lana_nlp.scripts.lyrics_analyzer import LyricsAnalyzer
from src.lana_nlp.scripts.text_cleaner import TextCleaner

# Load the lyrics csv
loader = LyricsDataLoader(
    Path("../data/raw/lyrics.csv")
)

df = loader.load()

print(
    df[
        df["lyrics"].apply(
            lambda x: not isinstance(x, str)
        )
    ]
)

# Clean text
cleaner = TextCleaner()

df["cleaned_lyrics"] = df["lyrics"].apply(
    cleaner.clean_text
)

df["tokens"] = df["cleaned_lyrics"].apply(
    cleaner.tokenize
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