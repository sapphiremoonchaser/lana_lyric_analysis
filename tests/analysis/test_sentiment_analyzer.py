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


