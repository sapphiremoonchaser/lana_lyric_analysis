import re
import pandas as pd


class TextCleaner:

    def __init__(self):
        pass

    def lowercase(
        self,
        text: str
    ) -> str:
        """
        Convert to lowercase.
        Args:
            text: text to convert

        Returns:
            The lowercase text
        """
        return text.lower()


    def remove_punctuation(
        self,
        text: str
    ) -> str:
        """
        Remove punctuation characters.

        Args:
            text: Text to remove characters from.

        Returns:
            The text string with punctuation removed.
        """
        return re.sub(
            r"[^\w\s]",
            "",
            text
        )

