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
from textstat import textstat
from textblob import TextBlob

import nltk
nltk.download(
    'vader_lexicon',
    'opinion_lexicon'
)
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon, words


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

        self.positive_words = set(
            opinion_lexicon.positive()
        )

        self.negative_words = set(
            opinion_lexicon.negative()
        )

        self.sia = SentimentIntensityAnalyzer()

        self._calculate_derived_columns()


    def _text_for_analysis(
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


    def _tokens_for_analysis(
        self,
        text: str | list
    ):
        """
        If lyrics are a string turn them to tokens.
        """
        if isinstance(text, list):
            return text

        if isinstance(text, str):
            return text.lower().split()

        return []


    def _use_tokenized_text(self) -> bool:
        non_null = self.df[self.text_column].dropna()

        if non_null.empty:
            return False

        return isinstance(non_null.iloc[0], list)


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


    def _calculate_reading_time(self):
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
        self.df["line_count"] = self.df[self.text_column].apply(
            lambda x: len(x.splitlines())
            if isinstance(x, str)
            else 0
        )


    def _calculate_derived_columns(self) -> None:
        """
        Calculate word_count, unique_word_count, and amount of time it
        takes to read based on a reading ability of 200 words per minute.
        Returns:
            None. Adds 3 columns to self.df.
        """

        self._calculate_word_count()
        self._calculate_unique_words()
        self._calculate_syllable_count()
        self._calculate_reading_time()
        self._calculate_line_count()
        self.sentiment_polarity()
        self.sentiment_subjectivity()
        self.positive_word_ratio()
        self.negative_word_ratio()


    def _words(self) -> list[str]:
        """
        Return all lyrics as a single list of lowercase words.

        This private helper method centralizes text extraction so that
        multiple analysis methods can reuse the same processing logic.

        Returns:
            A list containing every word from the song.
        """
        column = self.df[self.text_column]

        if self._use_tokenized_text():
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


    def _apply_textstat(
        self,
        func,
        column_name
    ):
        """
        Apply textstat library for readability metrics.
        Args:
            func: textstat function to apply
            column_name: output column
        """
        self.df[column_name] = self.df[self.text_column].appy(
            lambda x: func(x) if isinstance(x, str) else 0
        )


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
        stats = self.summary_by_album()

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

        if self._uses_tokenized_text():
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


    # ======================================================
    # Readability
    # ======================================================

    def flesch_reading_ease(self) -> None:
        """
        Calculate the flesch reading easy. It uses sentense length and syllables.

        Returns:
            None. Adds "flesch_reading_ease" to self.df.
        """
        self._apply_textstat(
            textstat.flesch_reading_ease,
            "flesch_reading_ease"
        )


    def flesch_kincaid(self) -> None:
        """
        Calculate the flesch kincaid reading ease.

        Returns:
            None. Adds "flesch_kincaid" to self.df.
        """
        self._apply_textstat(
            textstat.flesch_kincaid_grade(),
            "flesch_kincaid"
        )


    def gunning_fog(self) -> None:
        """
        Uses word complexity by number of syllables to calculate reading ease.
        Returns:
            None. Adds "gunning_fog" to self.df.
        """
        self._apply_textstat(
            textstat.gunning_fog,
            "gunning_fog"
        )


    def coleman_liau(self) -> None:
        """
        Uses average letters per word to calculate reading ease.

        Returns:
            None. Adds "coleman_liau" to self.df.
        """
        self._apply_textstat(
            textstat.coleman_liau,
            "coleman_liau"
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
                    self._text_for_analysis(x)
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
                    self._text_for_analysis(x)
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
            words = self._tokens_for_analysis(text)

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
            words = self._words_for_analysis(text)

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


