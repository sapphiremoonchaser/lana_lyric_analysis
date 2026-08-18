"""
Analyze song lyrics and compute readability metrics.

This module contains the ReadabilityAnalyzer class, which provides methods
for exploring different readability metrics.

The analyzer operates on a pandas DataFrame containing song metadata
and lyrics, and creates derived metrics.
"""
from textstat import textstat
import pandas as pd
from collections.abc import Callable

from lana_nlp.utils.text_utils import to_text


class ReadabilityAnalyzer:
    """
    Analyze a dataset of song lyrics for readability.

    This class provides methods for modeling readability using flesch reading ease,
    flesch kincaid, gunning fog.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "lyrics"
    ):
        self.df = df
        self.text_column = text_column


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
                    to_text(x)
                )
            )
        )


    def flesch_reading_ease(self) -> None:
        """
        Calculate the flesch reading easy. It uses sentence length and syllables.

        A lower flesch reading ease score means the text is more difficult.

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

        A lower flesch kincaid score means the text is more difficult.

        Returns:
            None. Adds "flesch_kincaid" to self.df.
        """
        self._apply_textstat(
            textstat.flesch_kincaid_grade,
            "flesch_kincaid"
        )


    def gunning_fog(self) -> None:
        """
        Uses word complexity by number of syllables to calculate reading ease.

        A higher score means the text is more difficult.

        Returns:
            None. Adds "gunning_fog" to self.df.
        """
        self._apply_textstat(
            textstat.gunning_fog,
            "gunning_fog"
        )


    def analyze(self) -> pd.DataFrame:
        """
        Calculate all readability metrics.

        Returns:
            DataFrame with readability metrics.
        """
        self.flesch_reading_ease()
        self.flesch_kincaid()
        self.gunning_fog()

        return self.df

