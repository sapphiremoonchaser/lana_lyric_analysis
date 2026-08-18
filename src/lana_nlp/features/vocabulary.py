"""
Analyze vocabulary habits in songs.

This module contains the VocabularyAnalyzer class, which provides methods
for exploring a song's vocabulary. It looks at word frequency, top words used,
and lexical diversity.

The VocabularyAnalyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics such as word count and
estimated reading time for each song.
"""
from collections import Counter

import pandas as pd


class VocabularyAnalyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics about the
    song's vocabulary, top words used, and lexical diversity.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "lyrics",
    ) -> None:
        self.df = df
        self.text_column = text_column


    def _get_song_words(
        self,
        text
    ) -> list[str]:
        """Return normalized words for a single song."""
        if isinstance(text, list):
            return [word.lower() for word in text]

        if isinstance(text, str):
            return text.lower().split()

        return text.lower().split()


    def song_unique_word_count(self, text) -> int:
        """Calculate vocabulary size for a single song."""
        words = self._get_song_words(text)

        return len(set(words))


    def song_lexical_diversity(self, text) -> float:
        """Calculate lexical diversity for a single song."""
        words = self._get_song_words(text)

        if not words:
            return 0.0

        return len(set(words)) / len(words)


    def song_average_word_length(self, text) -> float:
        """Calculate average word length for a single song."""
        words = self._get_song_words(text)

        if not words:
            return 0.0

        return sum(len(word) for word in words) / len(words)


    def analyze(self) -> None:
        """
        Add song level metrics to final analysis dataframe.
        """
        self.df["vocabulary_size"] = (
            self.df[self.text_column]
            .apply(self.song_unique_word_count)
        )

        self.df["lexical_diversity"] = (
            self.df[self.text_column]
            .apply(self.song_lexical_diversity)
        )

        self.df["average_word_length"] = (
            self.df[self.text_column]
            .apply(self.song_average_word_length)
        )









































