import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


class TextCleaner:

    def __init__(self):
        self.stop_words = set(
            stopwords.words('english')
        )

        self.lemmatizer = WordNetLemmatizer()


    def remove_annotations(
        self,
        text: str
    ) -> str:
        """
        Remove bracketed lyric annotations.

        Examples:
            [Verse 1]
            [Chorus]
            [Instrumental]
        """

        return re.sub(
            r"\[.*?\]",
            "",
            text
        )


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
        Normalize whitespace while preserving line breaks.
        """
        return "\n".join(
            " ".join(line.split()) for line in text.splitlines()
        )


    def tokenize(
        self,
        text: str
    ) -> list[str]:
        """
        Split text into a list of individual words.
        """
        return word_tokenize(text)


    def remove_stopwords(
        self,
        tokens: list[str]
    ):
        """
        Remove words like a and the.
        """

        return [
            word
            for word in tokens
            if word not in self.stop_words
        ]


    def lemmatize(
        self,
        tokens
    ):
        """
        Group similar words like love, loved, loving
        """

        return [
            self.lemmatizer.lemmatize(word)
            for word in tokens
        ]


    def basic_clean(
        self,
        text: str
    ) -> str:
        """
        Perform light cleaning while preserving lyrical structure.
        """

        # Deal with missing lyrics
        if pd.isna(text):
            return ""

        text = self.remove_annotations(text)
        text = self.remove_whitespace(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)

        return text.strip()


    def nlp_clean(
        self,
        text: str
    ) -> list[str]:
        """
        Perform aggresive NLP cleaning.

        Returns:
            List of cleaned tokens.
        """

        text = self.basic_clean(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return tokens