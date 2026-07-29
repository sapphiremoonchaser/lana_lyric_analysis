import re

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')


class TextCleaner:

    def __init__(self):
        pass

    def lowercase(
        self,
        text: str
    ) -> str:
        """
        Convert to lowercase.
        """
        return text.lower()


    def remove_punctuation(
        self,
        text: str
    ) -> str:
        """
        Remove punctuation characters.
        """
        return re.sub(
            r"[^\w\s]",
            "",
            text
        )

    def remove_whitespace(
        self,
        text: str
    ) -> str:
        """
        Normalize whitespace.
        """
        return " ".join(text.split())


    def clean_text(
        self,
        text: str
    ) -> str:
        """
        Apply all cleaning steps.

        Empty text appear as NaN is dealt with by converting it to "".
        """
        if not isinstance(
            text,
            str
        ):
            return ""

        text = self.lowercase(text)

        text = self.remove_punctuation(text)

        text = self.remove_whitespace(text)

        return text


    def tokenize(
        self,
        text: str
    ) -> list[str]:
        """
        Split text into a list of individual words.
        """
        return word_tokenize(text)
