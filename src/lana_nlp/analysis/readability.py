"""
Analyze song lyrics and compute readability metrics..

This module contains the ReadabilityAnalyzer class, which provides methods
for exploring different readability metrics..

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


class ReadabilityAnalyzer:
    """
    Analyze a dataset of song lyrics for readability..

    This class provides methods for modeling readability using flesch reading ease,
    flesch kincaid, gunning fog, and coleman liau.
    """

    def _apply_textstat(
        self,
        func,
        column_name
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
                if x
                else 0
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