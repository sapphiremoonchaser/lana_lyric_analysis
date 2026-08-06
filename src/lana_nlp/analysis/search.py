"""
Search for song lyrics.
"""
import pandas as pd

from lana_nlp.features.lyrics_features import LyricsFeatures
from lana_nlp.analysis.sentiment import SentimentAnalyzer
from lana_nlp.analysis.statistics import StatisticsAnalyzer
from lana_nlp.analysis.vocabulary import VocabularyAnalyzer
from lana_nlp.analysis.readability import ReadabilityAnalyzer


class LyricsAnalyzer:
    """
    Search for song lyrics.
    """

    def __init__(
        self,
        lyrics_df: pd.DataFrame,
        text_column: str = "lyrics"
    ):
        """
        Initialize the LyricsAnalyzer object.

        Creates a copy7 of the input DataFrame so the original data is not
        modified. It also calculates derived columns that are resued
        throughout the class.

        Args:
            lyrics_df: DataFrame containing song metadata and lyrics.
        """
        self.df = lyrics_df.copy()
        self.text_column = text_column


    def search(
        self,
        phrase: str
    ) -> pd.DataFrame:
        """
        Search song lyrics for a word or phrase.

        The search is case-insensitive and treats the search text
        as a literal string rather than a regular expression.

        Args:
            phrase: Word or phrase to search for.

        Returns:
            A DataFrame containing matching songs.
        """
        column = self.df[self.text_column]

        if self._use_tokenized_text():
            mask = column.apply(
                lambda x: (
                    phrase.lower() in " ".join(x).lower()
                    if isinstance(x, list)
                    else False
                )
            )

        else:
            mask = column.str.contains(
                phrase,
                case=False,
                regex=False,
                na=False
            )
        return self.df[mask]