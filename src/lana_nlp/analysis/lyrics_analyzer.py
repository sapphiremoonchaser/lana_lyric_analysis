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

from lana_nlp.features.lyrics_features import LyricsFeatures
from lana_nlp.analysis.sentiment import SentimentAnalyzer
from lana_nlp.analysis.statistics import StatisticsAnalyzer
from lana_nlp.analysis.vocabulary import VocabularyAnalyzer
from lana_nlp.analysis.readability import ReadabilityAnalyzer


class LyricsAnalyzer:
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


    def _calculate_derived_columns(self) -> None:
        """
        Calculate word_count, unique_word_count, and amount of time it
        takes to read based on a reading ability of 200 words per minute.
        Returns:
            None. Adds 3 columns to self.df.
        """

        calculations = LyricsFeatures(
            self.df,
            self.text_column
        )

        calculations.analyze()

        self.df = calculations.df


    def number_of_songs(self) -> int:
        """
        Return the total number of songs in the dataset.

        Returns:
            The total number of rows (songs).
        """
        if self.df.empty:
            return 0

        songs = self.df["song"].dropna()
        songs = songs[
            songs.str.strip() != "" # filter out songs that are just whitespace
        ]

        return len(songs)


    def albums(self) -> list[str]:
        """
        Alphabetical list of albums.

        Missing albums are removed before sorting.

        Returns:
            A list of unique album names.
        """
        if self.df.empty:
            return []

        albums = self.df["album"].dropna()
        albums = albums[
            albums.str.strip() != ""
        ]

        return sorted(albums)


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
            self.df["album"]
            .astype(str)
            .str.lower()
            .eq(album.lower())
        ].copy()


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


    def analyze(self) -> pd.DataFrame:
        """
        Calculate all features.
        """

        features = LyricsFeatures(
            self.df,
            text_column=self.text_column
        )
        self.df = features.analyze()

        self.statistics = StatisticsAnalyzer(self.df)
        self.vocabulary = VocabularyAnalyzer(self.df)
        self.sentiment = SentimentAnalyzer(self.df)
        self.readability = ReadabilityAnalyzer(self.df)

        return self.df



