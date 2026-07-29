import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Errors out without this
nltk.download('punkt_tab')


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
        text = self.remove_annotations(text)
        text = self.remove_whitespace(text)

        return text.strip()