"""
Create a pipeline to read a csv file containing lyrics, clean the data and analyze
the data.
"""
import pandas as pd

from lana_nlp.analysis.vocabulary import VocabularyAnalyzer
from lana_nlp.features.lyrics_features import LyricsFeatures
from lana_nlp.preprocessing.data_loader import LyricsDataLoader
from lana_nlp.preprocessing.text_cleaner import TextCleaner
from lana_nlp.analysis.readability import ReadabilityAnalyzer
from lana_nlp.analysis.sentiment import SentimentAnalyzer
from lana_nlp.analysis.vocabulary import VocabularyAnalyzer


def pipeline(
        filepath: str
) -> pd.DataFrame:
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
    features_analyzer = LyricsFeatures(
        df,
        text_column="basic_cleaned_lyrics"
    )

    features_analyzer.analyze()

    # Vocabulary features
    vocabulary_analyzer = VocabularyAnalyzer()

    df["lexical_diversity"] = (
        df["nlp_cleaned_lyrics"]
        .apply(vocabulary_analyzer.lexical_diversity)
    )

    df["vocabulary_size"] = (
        df["nlp_cleaned_lyrics"]
        .apply(vocabulary_analyzer.vocabulary_size)
    )

    df["word_frequency"] = (
        df["nlp_cleaned_lyrics"]
        .apply(vocabulary_analyzer.word_frequency())
    )

    df["unique_word_count"] = (
        df["nlp_cleaned_lyrics"]
        .apply(vocabulary_analyzer.unique_word_count())
    )

    # Add readability features
    readability_analyzer =  ReadabilityAnalyzer(
        df,
        text_column="basic_cleaned_lyrics"
    )

    readability_analyzer.analyze()

    # Add sentiment features
    sentiment_analyzer = SentimentAnalyzer(
        df,
        text_column="nlp_cleaned_lyrics"
    )

    sentiment_analyzer.analyze()

    df.to_csv("../data/processed/lyrics.csv")

    return df