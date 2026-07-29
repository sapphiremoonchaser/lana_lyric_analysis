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


