"""
Perform sentiment analysis on song lyrics..
"""
import pandas as pd
from collections import Counter

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from lana_nlp.utils.text_utils import (
    to_text,
    to_tokens
)


class SentimentAnalyzer:
    """
    Perform sentiment analysis on song lyrics.

    Measure polarity, subjectivity, and positive and negative sentiment ratios.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "lyrics"
    ):
        self.df = df
        self.text_column = text_column

        self.emotion_lexicon = self._load_emotion_lexicon()

        self.positive_words = set()
        self.negative_words = set()

        self.sia = SentimentIntensityAnalyzer()


    def _load_emotion_lexicon(self) -> dict:
        """
        Load NRC emotion lexicon.
        """
        lexicon_df = pd.read_csv(
            "data/raw/NRC-Emotion-Lexicon.csv"
        )

        lexicon = {}

        emotion_columns = [
            "Positive",
            "Negative",
            "Anger",
            "Anticipation",
            "Disgust",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
            "Trust"
        ]

        lexicon = {
            row["English (en)"]: [
                emotion
                for emotion in emotion_columns
                if row[emotion] == 1
            ]
            for _, row in lexicon_df.iterrows()
        }

        return lexicon


    def _calculate_emotions(
        self,
        words: list
    ) -> dict:
        """
        Calculate scores for the emotion lexicon.
        """
        EMOTIONS = [
            "Positive",
            "Negative",
            "Anger",
            "Anticipation",
            "Disgust",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
            "Trust"
        ]

        if isinstance(words, str):
            words = words.split()

        emotions = Counter()

        for word in words:
            if word in self.emotion_lexicon:
                for emotion in self.emotion_lexicon[word]:
                    emotions[emotion] += 1

        total = sum(emotions.values())

        if total == 0:
            return {}

        return {
            emotion: emotions[emotion] / total
            for emotion in EMOTIONS
        }


    def sentiment_polarity(self) -> None:
        """
        Calculate VADER sentiment polarity scores.

        Polarity ranges from:
            -1.0 = negative
            0.0 = neutral
            1.0 = positive

        Returns:
            None. Adds "sentiment_polarity" to self.df.
        """

        self.df["sentiment_polarity"] = self.df[self.text_column].apply(
            lambda x: (
                self.sia.polarity_scores(
                    to_text(x)
                )["compound"]
            )
        )


    def sentiment_subjectivity(self) -> None:
        """
        Calculate text subjectivity scores.

        Subjectivity ranges from:
            0.0 = objective
            1.0 = subjective

        Returns:
            None. Adds "subjectivity" to self.df.
        """

        self.df["subjectivity"] = self.df[self.text_column].apply(
            lambda x: (
                TextBlob(
                    to_text(x)
                ).sentiment.subjectivity
            )
        )


    def positive_word_ratio(self) -> None:
        """
        Calculate the ratio of positive words to total words.

        Returns a value between 0 and 1.
        Higher values indicate more positive language.
        """

        def calculate_ratio(text: str) -> float:
            words = to_tokens(text)

            if not words:
                return 0.0

            positive_count = sum(
                word in self.positive_words
                for word in words
            )

            return positive_count / len(words)

        self.df["positive_word_ratio"] = (
            self.df[self.text_column]
            .apply(calculate_ratio)
        )


    def negative_word_ratio(self) -> None:
        """
        Calculate the ratio of negative words.

        Returns a value between 0 and 1.
        Higher values indicate more negative language.
        """

        def calculate_ratio(text):
            words = to_tokens(text)

            if not words:
                return 0.0

            negative_count = sum(
                word in self.negative_words
                for word in words
            )

            return negative_count / len(words)

        self.df["negative_word_ratio"] = (
            self.df[self.text_column]
            .apply(calculate_ratio)
        )


    def emotion_scores(self) -> None:
        """
        Add NRC emotion scores to dataframe.
        """

        emotions = (
            self.df[self.text_column]
            .apply( # Calculate emotion columns
                lambda x: self._calculate_emotions(
                    to_tokens(x) # use tokens instead of string lyrics
                )
            )
        )

        emotion_df = pd.DataFrame(
            emotions.tolist()
        ).fillna(0)

        self.df = pd.concat(
            [
                self.df,
                emotion_df.add_prefix("emotion")
            ],
            axis=1
        )


    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate all sentiment columns.
        """

        self.sentiment_polarity()
        self.sentiment_subjectivity()
        self.emotion_scores()
        self.positive_word_ratio()
        self.negative_word_ratio()

        return self.df

