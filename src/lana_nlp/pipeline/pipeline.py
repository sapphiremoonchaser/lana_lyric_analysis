"""
Create a pipeline to read a csv file containing lyrics, clean the data and analyze
the data.
"""
import pandas as pd

from lana_nlp.features.basic_features import BasicFeatures
from lana_nlp.pipeline.data_loader import LyricsDataLoader
from lana_nlp.pipeline.text_cleaner import TextCleaner
from lana_nlp.features.readability import ReadabilityFeatures
from lana_nlp.features.sentiment import SentimentFeatures
from lana_nlp.features.vocabulary import VocabularyFeatures


def pipeline(
        filepath: str
) -> tuple[[pd.DataFrame], [pd.DataFrame]]:
    """
    Pipeline to read a csv file containing lyrics, clean the data and analyze. It
    gets dataframes for basic stats, readability, vocabulary, and sentiment.

    Args:
        filepath: the filepath of the csv file containing lyrics  to analyze

    Returns:
        One final dataframe containing all metrics
    """

    # Load the data
    loader = LyricsDataLoader(filepath=filepath)

    df = loader.load()

    # Text Cleaning and get tokens for nlp
    cleaner = TextCleaner()
    df["basic_cleaned_lyrics"] = df["lyrics"].apply(cleaner.basic_clean)
    df["nlp_cleaned_lyrics"] = df["lyrics"].apply(cleaner.nlp_clean)

    # Add basic features like word and line counts
    features_analyzer = BasicFeatures(
        df,
        text_column="basic_cleaned_lyrics"
    )

    features_analyzer.analyze()

    # Vocabulary features
    vocabulary_analyzer = VocabularyFeatures(
        df,
        text_column="nlp_cleaned_lyrics"
    )

    vocabulary_analyzer.analyze()

    # Add readability features
    readability_analyzer =  ReadabilityFeatures(
        df,
        text_column="basic_cleaned_lyrics"
    )

    readability_analyzer.analyze()

    # Add sentiment features
    sentiment_analyzer = SentimentFeatures(
        df,
        text_column="nlp_cleaned_lyrics"
    )

    sentiment_analyzer.analyze()

    # Save by song dataframe
    df.to_csv(
        "../data/processed/song_level_stats.csv",
        index=False
    )

    # Save by album dataframe
    emotion_columns = [
        column
        for column in df.columns
        if column.startswith("emotion_")
    ]

    emotion_stats = (
        df.groupby(["album", "year"])[emotion_columns]
        .mean()
        .reset_index()
    )

    album_stats = (
        df.groupby(["album", "year"])
        .agg(
            total_words=("word_count", "sum"),
            avg_words_per_song=("word_count", "mean"),
            vocabulary_size=("vocabulary_size", "mean"),
            lexical_diversity=("lexical_diversity", "mean"),
            average_word_length=("average_word_length", "mean"),
            flesch_reading_ease=("flesch_reading_ease", "mean"),
            flesch_kincaid=("flesch_kincaid", "mean"),
            gunning_fog=("gunning_fog", "mean"),
            sentiment_polarity=("sentiment_polarity", "mean"),
            subjectivity=("subjectivity", "mean"),
            positive_word_ratio=("positive_word_ratio", "mean"),
            negative_word_ratio=("negative_word_ratio", "mean"),
        )
        .reset_index()
    )

    # Add emotion statistics
    album_stats = album_stats.merge(
        emotion_stats,
        on=["album", "year"],
        how="left"
    )

    album_stats.to_csv("../data/processed/album_level_stats.csv", )

    return df, album_stats