"""
Analyze descriptive statistics for song lyrics.

This module contains the StatisticsAnalyzer class, which calculates aggregate
statistics across songs, albums, and release years.

Metrics include song length distributions, album summaries, ranking of songs by
length, and sentiment summaries.
"""
import pandas as pd


class StatisticsAnalyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics,
    searching lyrics, and measuring vocabulary usage across a collection
    of songs. Derived columns such as word_count and estimated reading
    time are calculated once during initialization.
    """
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df.copy()


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


    def summary_by_album(self) -> pd.DataFrame:
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


    def summary_by_year(self) -> pd.DataFrame:
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


    def longest_album(self) -> str:
        """
        Returns the longest album by word count.
        """
        return (
            self.summary_by_album()
            ["total_words"]
            .idxmax()
        )


    def average_album_sentiment(self) -> pd.Series:
        """
        Calculate average sentiment polarity by album.

        Returns:
            Mean sentiment score per album.
        """

        if "sentiment_polarity" not in self.df.columns:
            self.sentiment_polarity()

        sentiment = (
            self.df
            .dropna(subset=["album"])
            .groupby("album")["sentiment_polarity"]
            .mean()
        )

        return sentiment.fillna(0.0)


    def average_song_length_by_album(self) -> pd.DataFrame:
        """
        Returns the average length of songs by album by word count.
        """
        stats = self.summary_by_album()

        stats.reset_index(inplace=True)

        return stats[["album", "avg_words"]]


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