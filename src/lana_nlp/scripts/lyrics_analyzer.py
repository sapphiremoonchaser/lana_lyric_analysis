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
from functools import total_ordering
from itertools import count

import pandas as pd
from collections import Counter
from collections import defaultdict

from numpy.ma.extras import column_stack
from textstat import textstat
from textblob import TextBlob

import nltk
# nltk.download(
#     'vader_lexicon',
#     'opinion_lexicon'
# )
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon


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


    def _calculate_word_count(self) -> None:
        """
        Calculate word count by song.

        Returns:
            None. Adds "word_count" column to self.df.
        """
        column = self.df[self.text_column]

        # If the column is tokenized lyrics use this to calculate word_count
        if self._use_tokenized_text():
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


    def _calculate_unique_words(self) -> None:
        """
        Calculate unique words by song.

        Returns:
            None. Add "unique_words" column to self.df.
        """
        column = self.df[self.text_column]

        if self._use_tokenized_text():
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


    def _calculate_syllable_count(self) -> None:
        """
        Calculate the total number of syllables in each song.

        Returns:
            None. Adds "syllable_count" column to self.df.
        """
        column = self.df[self.text_column]

        if self._use_tokenized_text():
            self.df["syllable_count"] = column.apply(
                lambda x: sum(
                    textstat.syllable_count(word)
                    for word in x
                ) if isinstance(x, list) else 0
            )

        else:
            self.df["syllable_count"] = column.apply(
                lambda x: (
                    textstat.syllable_count(x)
                    if isinstance(x, str)
                    else 0
                )
            )


    def _calculate_reading_time(self) -> None:
        """
        Calculate the reading time based on 200 words per minute.

        Returns:
            None. Add "reading_minutes" to self.df.
        """
        # Estimate the reading time assuming an average reading speed
        # of 200 words per minute
        self.df["reading_minutes"] = self.df["word_count"] / 200


    def _calculate_line_count(self) -> None:
        """
        Calculate the number of lines per song.
        """
        self.df["line_count"] = (
            self.df[self.text_column]
            .apply(
                lambda x:
                    len(x.splitlines())
                if isinstance(x, str)
                    else len(x)
                if isinstance(x, list)
                    else 0
            )
        )



    # ======================================================
    # Dataset information
    # ======================================================


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


    def longest_album(self) -> str:
        """
        Returns the longest album by word count.
        """
        stats = self.summary_by_album()

        # Handle missing dataframe
        if stats.empty:
            return ""

        sorted_songs = stats.sort_values(
            by="total_words",
            ascending=False
        )

        return sorted_songs.index[0]


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
        stats = self.summary_by_album()

        stats.reset_index(inplace=True)

        return stats[["album", "avg_words"]]



    # ======================================================
    # Album Summary
    # ======================================================


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
                avg_reading_minutes=("reading_minutes", "mean"), # Average reading time
                avg_sentiment=("sentiment_polarity", "mean"), # Average sentiment
                avg_subjectivity=("subjectivity", "mean") # Average subjectivity
            )
            .sort_values(
                by="songs",
                ascending=False
            )
        )


    # ======================================================
    # Yearly Summary
    # ======================================================



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


    # ======================================================
    # Vocabulary
    # ======================================================



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



    # ======================================================
    # Sentiment
    # ======================================================

    def sentiment_polarity(self) -> None:
        """
        Calculate VADER sentiment polarity scores.

        The compound score ranges from -1 (negative) to 1 (positive).

        Returns:
            None. Adds "sentiment_polarity" to self.df.
        """

        self.df["sentiment_polarity"] = self.df[self.text_column].apply(
            lambda x: (
                self.sia.polarity_scores(
                    self._to_text(x)
                )["compound"]
            )
        )


    def sentiment_subjectivity(self) -> None:
        """
        Calculate text subjectivity scores.

        Subjectivity ranges from:
            0.0 = objective
            1.0 = subjective

        Returns:
            None. Adds "subjectivity" to self.df.
        """

        self.df["subjectivity"] = self.df[self.text_column].apply(
            lambda x: (
                TextBlob(
                    self._to_text(x)
                ).sentiment.subjectivity
            )
        )


    def positive_word_ratio(self) -> None:
        """
        Calculate the ratio of positive words to total words.

        Returns a value between 0 and 1.
        Higher values indicate more positive language.
        """

        def calculate_ratio(text: str) -> float:
            words = self._to_tokens(text)

            if not words:
                return 0.0

            positive_count = sum(
                word in self.positive_words
                for word in words
            )

            return positive_count / len(words)

        self.df["positive_word_ratio"] = (
            self.df[self.text_column]
            .apply(calculate_ratio)
        )


    def negative_word_ratio(self) -> None:
        """
        Calculate the ratio of negative words.

        Returns a value between 0 and 1.
        Higher values indicate more negative language.
        """

        def calculate_ratio(text):
            words = self._to_tokens(text)

            if not words:
                return 0.0

            negative_count = sum(
                word in self.negative_words
                for word in words
            )

            return negative_count / len(words)

        self.df["negative_word_ratio"] = (
            self.df[self.text_column]
            .apply(calculate_ratio)
        )


    # ======================================================
    # Emotions
    # ======================================================

