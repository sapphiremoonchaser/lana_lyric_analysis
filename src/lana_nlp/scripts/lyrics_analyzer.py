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
from collections import Counter

from pandas.core.interchange import column


class LyricsAnalyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics,
    searching lyrics, and measuring vocabulary usage across a collection
    of songs. Derived columns such as word_count and estimated reading
    time are calculated once during initialization.
    """

    # ======================================================
    # Initialization
    # ======================================================


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

        self._calculate_derived_columns()


    def _calculate_derived_columns(self) -> None:
        """
        Calculate word_count, unique_word_count, and amount of time it
        takes to read based on a reading ability of 200 words per minute.
        Returns:
            None. Adds 3 columns to self.df.
        """

        column = self.df[self.text_column]

        # Handle edge case where are lyrics are NA
        non_null = column.dropna()

        if non_null.empty:
            self.df["word_count"] = 0

        # If the column is a list use this method to calculate word count
        elif isinstance(non_null.iloc[0], list):
            self.df["word_count"] = column.apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        # If the column is a string use this method to calculate word_count
        else:
            self.df["word_count"] = (
                column
                .fillna("")
                .str.split()
                .str.len()
            )

        # Get unique words
        if non_null.empty:
            self.df["unique_words"] = 0

        elif isinstance(non_null.iloc[0], list):
            self.df["unique_words"] = column.apply(
                lambda x: len(set(x))
                if isinstance(x, list)
                else 0
            )

        else:
            self.df["unique_words"] = column.apply(
                lambda x: (
                    len(set(x.split()))
                    if isinstance(x, str)
                    else 0
                ))

        # Estimate the reading time assuming an average reading speed
        # of 200 words per minute
        self.df["reading_minutes"] = self.df["word_count"] / 200


    def _words(self) -> list[str]:
        """
        Return all lyrics as a single list of lowercase words.

        This private helper method centralizes text extraction so that
        multiple analysis methods can reuse the same processing logic.

        Returns:
            A list containing every word from the song.
        """
        column = self.df[self.text_column]

        non_null = column.dropna()

        if non_null.empty:
            return []

        if isinstance(non_null.iloc[0], list):
            words = []

            for tokens in column:
                if isinstance(tokens, list):
                    words.extend(tokens)

            return [word.lower() for word in words]

        return (
            " ".join(column.fillna(""))
            .lower()
            .split()
        )


    def _word_counter(self) -> Counter:
        """
        Return word frequencies across all lyrics.
        """
        return Counter(self._words())


    # ======================================================
    # Dataset information
    # ======================================================


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


    def number_of_songs_by_album(self) -> pd.Series:
        """
        Count the number of songs on each album.

        Groups the dataset by album and counts how many songs belong
        to each one.

        Returns:
            A pandas Series whose index contains album names and whose
            values contain the number of songs.
        """
        return (
            self.df
            .groupby("album")       # Group by album
            .size()                 # Count songs in each group
            .sort_values(ascending=False) # Show largest albums first
        )


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


    def longest_album(self) -> str:
        """
        Returns the longest album by word count.
        """
        stats = self.album_summary()

        # Handle missing dataframe
        if stats.empty:
            return ""

        sorted_songs = stats.sort_values(
            by="total_words",
            ascending=False
        )

        return sorted_songs.index[0]


    # ======================================================
    # Song statistics
    # ======================================================


    def song_length_stats(self) -> dict[str, float]:
        """
        Calculate descriptive statistics for songs length.

        Uses the precomputed word counts created during initialization.

        Returns:
            A dictionary containing the mean, median, minimum,
            and maximum song lengths in words.
        """
        stats = self.df["word_count"].agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )

        stats = {
            key: 0 if pd.isna(value) else value
            for key, value in stats.items()
        }

        return stats


    def longest_songs(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Returns the longest songs by word count.

        Args:
            n: number of songs to return

        Returns:
            A DataFrame containing the longest songs sorted by
            descending word count.
        """

        return (
            self.df
            .sort_values(
                by=["word_count"],
                ascending=False
            )
            .head(n)
        )


    def shortest_songs(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Returns the shorted songs by word count.

        Args:
            n: number of songs to return

        Returns:
            A DataFrame containing the shorted songs sorted by
            ascending word count.
        """
        return (
            self.df
            .sort_values("word_count")
            .head(n)
        )


    def average_song_length_by_album(self) -> pd.DataFrame:
        """
        Returns the average length of songs by album by word count.
        """
        stats = self.album_summary()

        stats.reset_index(inplace=True)

        return stats[["album", "avg_words"]]


    def line_count(self) -> None:
        """
        Calculate the number of lines per song.
        """
        self.df["line_count"] = self.df[self.text_column].apply(
            lambda x: len(x.splitlines())
            if isinstance(x, str)
            else 0
        )


    # ======================================================
    # Album Summary
    # ======================================================


    def album_summary(self) -> pd.DataFrame:
        """
        Calculated summary statistics for each album.

        Statistics include:
            - number of songs
            - average words per song
            - total words on the album

        Returns:
            A DataFrame containing one row per album.
        """
        return (
            self.df
            .groupby("album")
            .agg(
                songs=("song", "count"),           # Number of songs
                avg_words=("word_count", "mean"),   # Average song length
                median_words=("word_count", "median"), # Median number of words
                min_words=("word_count", "min"),
                max_words=("word_count", "max"),
                total_words=("word_count", "sum"),   # Total words
                avg_reading_minutes=("reading_minutes", "mean") # Average reading time
            )
            .sort_values(
                by="songs",
                ascending=False
            )
        )


    # ======================================================
    # Yearly Summary
    # ======================================================


    def yearly_summary(self) -> pd.DataFrame:
        """
        Calculated summary statistics for each year.
        """
        return (
            self.df
            .groupby("year")
            .agg(
                songs=("song", "count"),
                avg_words=("word_count", "mean"),
                median_words=("word_count", "median"),
                min_words=("word_count", "min"),
                max_words=("word_count", "max"),
                total_words=("word_count", "sum"),
                avg_reading_minutes=("reading_minutes", "mean") # Average reading time
            )
        )


    # ======================================================
    # Search
    # ======================================================


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

        non_null = column.dropna()

        if non_null.empty:
            return self.df.iloc[0:0]

        if isinstance(non_null.iloc[0], list):
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


    # ======================================================
    # Vocabulary
    # ======================================================


    def top_n_words(
        self,
        n: int = 25
    ) -> list[tuple[str, int]]:
        """
        Returns the most frequently occurring words.

        Args:
            n: Number of words to return.

        Returns:
            A list of (word, frequency) tuples sorted from
            most common to least common.
        """
        return self.word_frequency().most_common(n)


    def average_word_length(self) -> float:
        """
        Calculate the average word length across all lyrics.

        Returns:
            The mean number of characters per word.
        """
        words = self._words()

        if not words:
            return 0.0

        return (
            sum(len(word) for word in words)
            / len(words)
        )


    def word_frequency(self) -> Counter:
        """
        Calculate how many times each word appears across all lyrics.

        Returns:
            Counter mapping each word to the number of times it appears across the
            dataset.
        """
        return self._word_counter()


    def vocabulary_size(self) -> int:
        """
        Calculate the size of the vocabulary.

        Vocabulary size is the number of unique words that appear
        across all lyrics.

        Returns:
            The number of distinct words.
        """
        words = self._words()

        # Convert to a set to remove duplicates
        return len(set(words))


    def lexical_diversity(self) -> float:
        """
        Calculate lexical diversity.

        Lexical diversity is the ratio of unique words to the total
        number of words. Higher values indicate a more varied vocabulary.

        Returns:
            A value between 0 and 1 representing vocabulary diversity.
        """
        words = self._words()

        # Avoid division by zero if there are no lyrics.
        if not words:
            return 0.0

        # unique words / total number of words
        return len(set(words)) / len(words)


    def unique_words(self) -> None:
        """
        Calculate the number of unique words.
        """

        column = self.df[self.text_column]

        non_null = column.dropna()

        if non_null.empty:
            self.df["unique_words"] = 0

        elif isinstance(non_null.iloc[0], list):
            self.df["unique_words"] = column.apply(
                lambda x: len(set(x)) if isinstance(x, list) else 0
            )

        else:
            self.df["unique_words"] = column.apply(
                lambda x: len(set(x.split())) if isinstance(x, str) else 0
            )

