"""
Analyze vocabulary habits in songs..

This module contains the VocabularyAnalyzer class, which provides methods
for exploring a song's vocabulary. It looks at word frequency, top words used,
and lexical diversity.

The VocabularyAnalyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics such as word count and
estimated reading time for each song.
"""
import re
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


    def _normalize_word(
        self,
        word: str,
    ):
        """
        Remove punctuation and lowercase.
        """
        return re.sub(
            r"[^\w']",
            "",
            word.lower(),
        )


    def _get_words(self) -> list[str]:
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


    def word_frequency(self) -> Counter:
        """
        Calculate how many times each word appears across all lyrics.

        Returns:
            Counter mapping each word to the number of times it appears across the
            dataset.
        """
        return Counter(self._get_words())


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


    def unique_word_count(self) -> int:
        """
        Calculate the size of the vocabulary.

        Vocabulary size is the number of unique words that appear
        across all lyrics.

        Returns:
            The number of distinct words.
        """
        words = self._get_words()

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
        words = self._get_words()

        # Avoid division by zero if there are no lyrics.
        if not words:
            return 0.0

        # unique words / total number of words
        return len(set(words)) / len(words)


    def average_word_length(self) -> float:
        """
        Calculate the average word length across all lyrics.

        Returns:
            The mean number of characters per word.
        """
        words = self._get_words()

        if not words:
            return 0.0

        return (
            sum(len(word) for word in words)
            / len(words)
        )
