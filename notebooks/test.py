import pandas as pd
from pathlib import Path

from lana_nlp.analysis.sentiment import SentimentAnalyzer
from lana_nlp.preprocessing.data_loader import LyricsDataLoader
from lana_nlp.preprocessing.text_cleaner import TextCleaner

loader = LyricsDataLoader(
    Path("../data/raw/lyrics.csv")
)

df = loader.load()

cleaner = TextCleaner()

df["basic_cleaned_lyrics"] = df["lyrics"].apply(cleaner.basic_clean)

df["nlp_cleaned_lyrics"] = df["basic_cleaned_lyrics"].apply(cleaner.nlp_clean)

sentiment_analyzer = SentimentAnalyzer(
    df,
    text_column="nlp_cleaned_lyrics",
)

x = 1