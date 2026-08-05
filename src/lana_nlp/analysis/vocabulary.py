"""
Analyze vocabulary habits in songs..

This module contains the VocabularyAnalyzer class, which provides methods
for exploring a song's vocabulary. It looks at word frequency, top words used,
and lexical diversity.

The VocabularyAnalyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics such as word count and
estimated reading time for each song.
"""
from collections import Counter


class LyricsAnalyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics about the
    song's vocabulary, top words used, and lexical diversity.
    """

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


    def word_frequency(self) -> Counter:
        """
        Calculate how many times each word appears across all lyrics.

        Returns:
            Counter mapping each word to the number of times it appears across the
            dataset.
        """
        return self._word_counter()


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
