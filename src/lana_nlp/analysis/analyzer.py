"""
Analyze song lyrics and compute descriptive statistics.

This module contains the LyricsAnalyzer class, which provides methods
for exploring a collection of song lyrics. It calculates song and album
statistics, search lyrics, measures vocabulary usage, and computes
basic NLP metrics such as lexical diversity and word frequency.

The analyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics such as word count and
estimated reading time for each song.
"""
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon

from lana_nlp.features.feature_engineering import LyricsFeatures

class Analyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics,
    searching lyrics, and measuring vocabulary usage across a collection
    of songs. Derived columns such as word_count and estimated reading
    time are calculated once during initialization.
    """

    def __init__(
        self,
        lyrics_df: pd.DataFrame,
        text_column: str = "basic_cleaned_lyrics"
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

        self.positive_words = set(
            opinion_lexicon.positive()
        )

        self.negative_words = set(
            opinion_lexicon.negative()
        )

        self.sia = SentimentIntensityAnalyzer()

        self.emotion_lexicon = self._load_emotion_lexicon()

        self._calculate_derived_columns()


    def _calculate_derived_columns(self) -> None:
        """
        Calculate word_count, unique_word_count, and amount of time it
        takes to read based on a reading ability of 200 words per minute.
        Returns:
            None. Adds 3 columns to self.df.
        """

        calculations = LyricsFeatures(self.df)

        calculations._calculate_word_count()
        calculations._calculate_unique_words()
        calculations._calculate_syllable_count()
        calculations._calculate_reading_time()
        calculations._calculate_line_count()
        calculations.sentiment_polarity()
        calculations.sentiment_subjectivity()
        calculations.positive_word_ratio()
        calculations.negative_word_ratio()


    def _to_text(
        self,
        text
    ) -> str:
        """
        If lyrics are tokens turn them into a string.
        """

        if isinstance(text, list):
            return " ".join(text)

        if isinstance(text, str):
            return text

        return ""


    def _to_tokens(
        self,
        text: str | list
    ) -> list[str]:
        """
        If lyrics are a string turn them to tokens.
        """
        if isinstance(text, list):
            return text

        if isinstance(text, str):
            return text.lower().split()

        return []



    def number_of_songs(self) -> int:
        """
        Return the total number of songs in the dataset.

        Returns:
            The total number of rows (songs).
        """
        return len(self.df)


    def albums(self) -> list[str]:
        """
        Alphabetical list of albums.

        Missing albums are removed before sorting.

        Returns:
            A list of unique album names.
        """
        return sorted(self.df["album"].dropna().unique())


    def songs_by_album(
        self,
        album: str
    ) -> pd.DataFrame:
        """
        Creates dataframe of songs appearing on a specified album.

        Args:
            album: album you want to see songs for

        Returns:
            DataFrame with songs from specified album.
        """
        return self.df[
            self.df["album"].fillna("").str.lower() == album.lower()
        ]


    def songs_by_year(
        self,
        year: int
    ) -> pd.DataFrame:
        """
        Creates dataframe of songs released in specified year.

        Args:
            year: year you want to see songs for

        Returns:
            DataFrame with songs from specified year.
        """
        return self.df[
            self.df["year"] == year
        ]


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
