# Imports
from unittest.mock import patch
import pandas as pd

from lana_nlp.analysis.sentiment import SentimentAnalyzer


def test_load_emotion_lexicon_returns_dict():
    mock_lexicon = pd.DataFrame(
        {
            "English (en)": [
                "happy",
                "sad",
                "angry"
            ],
            "Positive": [1, 0, 0],
            "Negative": [0, 1, 0],
            "Anger": [0, 0, 1],
            "Joy": [1, 0, 0],
            "Fear": [0, 0, 0],
            "Trust": [1, 0, 0],
            "Anticipation": [0, 0, 0],
            "Disgust": [0, 0, 0],
            "Sadness": [0, 1, 0],
            "Surprise": [0, 0, 0],
        }
    )

    with patch("pandas.read_csv", return_value=mock_lexicon):
        analyzer = SentimentAnalyzer(
            pd.DataFrame(),
            lexicon_path="fake_path.csv"
        )

        lexicon = analyzer._load_emotion_lexicon()

    assert isinstance(lexicon, dict)

    assert lexicon["happy"] ==[
        "Positive",
        "Joy",
        "Trust"
    ]

    assert lexicon["sad"] ==[
        "Negative",
        "Sadness",
    ]

    assert lexicon["angry"] == [
        "Anger"
    ]


def test_calculate_emotions_returns_normalized_scores():
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.emotion_lexicon = {
        "happy": ["Positive", "Joy"],
        "sad": ["Negative", "Sadness"],
    }

    result = analyzer._calculate_emotions(
        ["happy", "sad"]
    )

    assert result["Positive"] == 0.25
    assert result["Joy"] == 0.25
    assert result["Negative"] == 0.25
    assert result["Sadness"] == 0.25


def test_calculate_emotions_accepts_string():
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        "happy"
    )

    assert result["Joy"] == 1.0


def test_calculate_emotions_returns_empty_dict_when_no_matches():
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        ["unknown"]
    )

    assert result == {}


def test_calculate_emotions_ignores_unknown_words():
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        ["happy", "unknown"]
    )

    assert result["Joy"] == 1.0


def test_sentiment_polarity_structure_and_value(sample_df):
    analyzer = SentimentAnalyzer(sample_df, text_column="lyrics")

    analyzer.sentiment_polarity()

    scores = analyzer.df["sentiment_polarity"]

    # Test structure
    assert "sentiment_polarity" in analyzer.df.columns
    assert len(analyzer.df["sentiment_polarity"]) == len(sample_df)

    # Test that numeric values are actually calculated
    assert scores.notna().all()
    assert pd.api.types.is_numeric_dtype(scores)

    # Test values are within VADER ranges
    assert scores.between(-1, 1).all()


def test_sentiment_polarity_handles_tokens() -> None:
    df = pd.DataFrame({
        "lyrics": [
            ["happy", "love", "beautiful"],
            ["hate", "sad", "angry"]
        ]
    })

    analyzer = SentimentAnalyzer(df, text_column="lyrics")

    analyzer.sentiment_polarity()

    assert analyzer.df["sentiment_polarity"].notna().all()


def test_sentiment_subjectivity_structure_and_value(sample_df):
    analyzer = SentimentAnalyzer(sample_df, text_column="lyrics")

    analyzer.sentiment_subjectivity()

    scores = analyzer.df["subjectivity"]

    # Test structure
    assert "subjectivity" in analyzer.df.columns
    assert len(analyzer.df["subjectivity"]) == len(sample_df)

    # Test that numeric values are actually calculated
    assert scores.notna().all()
    assert pd.api.types.is_numeric_dtype(scores)

    # Test values are within VADER ranges
    assert scores.between(0, 1).all()


def test_sentiment_subjectivity_handles_tokens() -> None:
    df = pd.DataFrame({
        "lyrics": [
            ["happy", "love", "beautiful"],
            ["hate", "sad", "angry"]
        ]
    })

    analyzer = SentimentAnalyzer(df, text_column="lyrics")

    analyzer.sentiment_subjectivity()

    assert analyzer.df["subjectivity"].notna().all()


def test_positive_word_ratio_structure(sample_df) -> None:
    analyzer = SentimentAnalyzer(sample_df, text_column="lyrics")

    analyzer.positive_word_ratio()

    ratios = analyzer.df["positive_word_ratio"]

    assert "positive_word_ratio" in analyzer.df.columns
    assert ratios.between(0, 1).all()


def test_positive_word_ratio_value() -> None:
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.df = pd.DataFrame({
        "lyrics": ["happy love sad"]
    })

    analyzer.text_column = "lyrics"
    analyzer.positive_words = {"happy", "love"}

    analyzer.positive_word_ratio()

    assert analyzer.df.loc[0, "positive_word_ratio"] == 2 / 3


def test_positive_word_ratio_returns_zero_for_empty_text():
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)

    analyzer.df = pd.DataFrame({
        "lyrics": [""]
    })
    analyzer.text_column = "lyrics"
    analyzer.positive_words = {"happy"}

    analyzer.positive_word_ratio()

    assert analyzer.df.loc[0, "positive_word_ratio"] == 0.0


