"""
Calculate derived metrics around words, line count, and reading time.
"""
from textstat import textstat
import pandas as pd


class LyricsFeatures:
    """
    Calculate word count, syllable count, line count, and reading time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str
    ):
        self.df = df
        self.text_column = text_column


    def _use_tokenized_text(self) -> bool:
        non_null = self.df[self.text_column].dropna()

        if non_null.empty:
            return False

        return isinstance(non_null.iloc[0], list)


    def calculate_word_count(self) -> None:
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


    def calculate_unique_words(self) -> None:
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


    def calculate_syllable_count(self) -> None:
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


    def calculate_reading_time(self) -> None:
        """
        Calculate the reading time based on 200 words per minute.

        Returns:
            None. Add "reading_minutes" to self.df.
        """
        # Estimate the reading time assuming an average reading speed
        # of 200 words per minute
        self.df["reading_minutes"] = self.df["word_count"] / 200


    def calculate_line_count(self) -> None:
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


    def calculate_all(self) -> pd.DataFrame:
        """Calculate all derived features."""
        self.calculate_word_count()
        self.calculate_unique_words()
        self.calculate_syllable_count()
        self.calculate_line_count()
        self.calculate_reading_time()

        return self.df