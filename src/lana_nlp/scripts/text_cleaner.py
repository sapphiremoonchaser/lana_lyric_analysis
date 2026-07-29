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
        return "".join(text.split())


    def clean_text(
        self,
        text: str
    ) -> str:
        """
        Apply all cleaning steps.
        """

        text = self.lowercase(text)

        text = self.remove_punctuation(text)

        text = self.remove_whitespace(text)

        return text


