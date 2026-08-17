"""
Create a pipeline to read a csv file containing lyrics, clean the data and analyze
the data.
"""
import pandas as pd

from lana_nlp.preprocessing.data_loader import LyricsDataLoader
from lana_nlp.preprocessing.text_cleaner import TextCleaner

def pipeline(
        filepath: str
) -> pd.DataFrame:
    """
    Pipeline to read a csv file containing lyrics, clean the data and analyze

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

    return df