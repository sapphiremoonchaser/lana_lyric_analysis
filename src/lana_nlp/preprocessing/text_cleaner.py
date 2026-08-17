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

        self.custom_stopwords = {
            "dont",
            "im",
            "na",
            "youre",
            "like",
            "oh",
            "ahahahahah",
            "yeah",
            "ah",
            "ooh",
            "lala",
            "get",
            "let",
            "ha",
            "wan"
        }


        self.lemmatizer = WordNetLemmatizer()

    def expand_contractions(
            self,
            text: str
    ) -> str:
        """
        Expand common English contractions.
        """

        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "I'm": "I am",
            "I've": "I have",
            "I'll": "I will",
            "I'd": "I would",
            "you're": "you are",
            "you've": "you have",
            "you'll": "you will",
            "you'd": "you would",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "we've": "we have",
            "they're": "they are",
            "they've": "they have",
        }

        for contraction, expanded in contractions.items():
            text = re.sub(
                rf"\b{re.escape(contraction)}\b",
                expanded,
                text,
                flags=re.IGNORECASE
            )

        return text


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
        all_stopwords = self.stop_words | self.custom_stopwords

        return [
            word
            for word in tokens
            if word not in all_stopwords
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
        text = self.expand_contractions(text)
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

        # text = text.replace("[", "")
        # text = text.replace("]", "")
        # text = text.replace("'", "")

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return tokens