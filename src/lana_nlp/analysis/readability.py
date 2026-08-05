"""
Analyze song lyrics and compute readability metrics..

This module contains the ReadabilityAnalyzer class, which provides methods
for exploring different readability metrics..

The analyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics such as word count and
estimated reading time for each song.
"""
from textstat import textstat
import pandas as pd
from collections.abc import Callable


class ReadabilityAnalyzer:
    """
    Analyze a dataset of song lyrics for readability..

    This class provides methods for modeling readability using flesch reading ease,
    flesch kincaid, gunning fog, and coleman liau.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "lyrics"
    ):
        self.df = df
        self.text_column = text_column


    def _to_text(
        self,
        text: str | list[str] | None
    ) -> str:
        """
        Helper fuctions to covert tokens to a string.

        Args:
            text: lyrics or list of lyrics.
        """
        if isinstance(text, list):
            return " ".join(text)

        if isinstance(text, str):
            return text

        return ""


    def _apply_textstat(
        self,
        func: Callable[[str], float],
        column_name: str
    ) -> None:
        """
        Apply textstat library for readability metrics.
        Args:
            func: textstat function to apply
            column_name: output column
        """
        self.df[column_name] = (
            self.df[self.text_column]
            .apply(
                lambda x: func(
                    self._to_text(x)
                )
            )
        )


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