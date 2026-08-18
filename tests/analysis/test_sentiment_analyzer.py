# Imports
from unittest.mock import patch
import pandas as pd
import pytest

from lana_nlp.features.sentiment import SentimentFeatures


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
        analyzer = SentimentFeatures(
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


def test_load_sentiment_words_creates_sets():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.emotion_lexicon = {
        "happy": ["Positive", "Joy"],
        "sad": ["Negative", "Sadness"],
        "love": ["Positive"]
    }

    analyzer._load_sentiment_words()

    assert analyzer.positive_words == {
        "happy",
        "love"
    }

    assert analyzer.negative_words == {
        "sad"
    }


def test_calculate_emotions_returns_normalized_scores():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

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
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        "happy"
    )

    assert result["Joy"] == 1.0


def test_calculate_emotions_returns_empty_dict_when_no_matches():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        ["unknown"]
    )

    assert result == {}


def test_calculate_emotions_ignores_unknown_words():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.emotion_lexicon = {
        "happy": ["Joy"]
    }

    result = analyzer._calculate_emotions(
        ["happy", "unknown"]
    )

    assert result["Joy"] == 1.0


def test_sentiment_polarity_structure_and_value(sample_df):
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

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

    analyzer = SentimentFeatures(df, text_column="lyrics")

    analyzer.sentiment_polarity()

    assert analyzer.df["sentiment_polarity"].notna().all()


def test_sentiment_subjectivity_structure_and_value(sample_df):
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

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

    analyzer = SentimentFeatures(df, text_column="lyrics")

    analyzer.sentiment_subjectivity()

    assert analyzer.df["subjectivity"].notna().all()


def test_positive_word_ratio_structure(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    analyzer.positive_word_ratio()

    ratios = analyzer.df["positive_word_ratio"]

    assert "positive_word_ratio" in analyzer.df.columns
    assert ratios.between(0, 1).all()


def test_positive_word_ratio_value() -> None:
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.df = pd.DataFrame({
        "lyrics": ["happy love sad"]
    })

    analyzer.text_column = "lyrics"
    analyzer.positive_words = {"happy", "love"}

    analyzer.positive_word_ratio()

    assert analyzer.df.loc[0, "positive_word_ratio"] == 2 / 3


def test_positive_word_ratio_returns_zero_for_empty_text():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.df = pd.DataFrame({
        "lyrics": [""]
    })
    analyzer.text_column = "lyrics"
    analyzer.positive_words = {"happy"}

    analyzer.positive_word_ratio()

    assert analyzer.df.loc[0, "positive_word_ratio"] == 0.0


def test_negative_word_ratio_structure(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    analyzer.negative_word_ratio()

    ratios = analyzer.df["negative_word_ratio"]

    assert "negative_word_ratio" in analyzer.df.columns
    assert ratios.between(0, 1).all()


def test_negative_word_ratio_value() -> None:
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.df = pd.DataFrame({
        "lyrics": ["happy love sad"]
    })

    analyzer.text_column = "lyrics"
    analyzer.negative_words = {"sad"}

    analyzer.negative_word_ratio()

    assert analyzer.df.loc[0, "negative_word_ratio"] == 1 / 3


def test_negative_word_ratio_returns_zero_for_empty_text():
    analyzer = SentimentFeatures.__new__(SentimentFeatures)

    analyzer.df = pd.DataFrame({
        "lyrics": [""]
    })
    analyzer.text_column = "lyrics"
    analyzer.negative_words = {"happy"}

    analyzer.negative_word_ratio()

    assert analyzer.df.loc[0, "negative_word_ratio"] == 0.0


def test_emotion_scores_adds_emotion_columns(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    analyzer.emotion_scores()

    expected_columns = [
        "emotionPositive",
        "emotionNegative",
        "emotionAnger",
        "emotionAnticipation",
        "emotionDisgust",
        "emotionFear",
        "emotionJoy",
        "emotionSadness",
        "emotionSurprise",
        "emotionTrust",
    ]

    for column in expected_columns:
        assert column in analyzer.df.columns


def test_emotion_scores_calculates_values() -> None:
    df = pd.DataFrame({
        "lyrics": [
            "happy sad"
        ]
    })

    analyzer = SentimentFeatures.__new__(SentimentFeatures)
    analyzer.df = df.copy()
    analyzer.text_column = "lyrics"
    analyzer.emotion_lexicon = {
        "happy": ["Positive", "Joy"],
        "sad": ["Negative", "Sadness"],
    }

    analyzer.emotion_scores()

    assert analyzer.df.loc[0, "emotionPositive"] == 0.25
    assert analyzer.df.loc[0, "emotionJoy"] == 0.25
    assert analyzer.df.loc[0, "emotionNegative"] == 0.25
    assert analyzer.df.loc[0, "emotionSadness"] == 0.25


def test_emotion_scores_fills_missing_emotions_with_zero() -> None:
    df = pd.DataFrame({
        "lyrics": ["happy"]
    })

    analyzer = SentimentFeatures.__new__(SentimentFeatures)
    analyzer.df = df.copy()
    analyzer.text_column = "lyrics"
    analyzer.emotion_lexicon = {
        "happy": ["Positive", "Joy"],
    }

    analyzer.emotion_scores()

    assert analyzer.df.loc[0, "emotionPositive"] == 0.5
    assert analyzer.df.loc[0, "emotionJoy"] == 0.5
    assert analyzer.df.loc[0, "emotionNegative"] == 0
    assert analyzer.df.loc[0, "emotionAnger"] == 0


def test_emotion_scores_preserves_original_columns(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    original_columns = sample_df.columns.tolist()

    analyzer.emotion_scores()

    for column in original_columns:
        assert column in analyzer.df.columns


def test_emotion_scores_returns_zero_for_no_matches() -> None:
    df = pd.DataFrame({
        "lyrics": ["xyzabc"]
    })

    analyzer = SentimentFeatures.__new__(SentimentFeatures)
    analyzer.df = df.copy()
    analyzer.text_column = "lyrics"
    analyzer.emotion_lexicon = {
        "happy": ["Positive", "Joy"],
    }

    analyzer.emotion_scores()

    assert analyzer.df.loc[0, "emotionPositive"] == 0
    assert analyzer.df.loc[0, "emotionJoy"] == 0


def test_calculate_all_structure_and_values(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    result = analyzer.analyze()

    # Check structure
    assert isinstance(result, pd.DataFrame)

    # Check for correct columns added
    assert "sentiment_polarity" in result.columns
    assert "subjectivity" in result.columns
    assert "positive_word_ratio" in result.columns
    assert "negative_word_ratio" in result.columns
    assert "emotionPositive" in result.columns


def test_average_album_sentiment_structure(sample_df) -> None:
    analyzer = SentimentFeatures(sample_df, text_column="lyrics")

    result = analyzer.average_album_sentiment()

    assert isinstance(result, pd.Series)


def test_average_album_sentiment_ignores_missing_albums():
    df = pd.DataFrame({
        "album": ["NFR", None, "NFR"],
        "sentiment_polarity": [0.5, 0.9, -0.1],
    })

    analyzer = SentimentFeatures(df, text_column="lyrics")

    result = analyzer.average_album_sentiment()

    assert list(result.index) == ["NFR"]
    assert result["NFR"] == pytest.approx(0.2)



