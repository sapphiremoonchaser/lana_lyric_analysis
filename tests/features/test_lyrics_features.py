import pandas as pd
import numpy as np

from lana_nlp.analysis.basic_features import FeaturesAnalyzer


def test_use_tokenized_text_returns_false_for_strings():
    df = pd.DataFrame({
        "lyrics": ["love forever", "dream ocean"]
    })

    analyzer = FeaturesAnalyzer(df, text_column="lyrics")

    assert analyzer._use_tokenized_text() is False


def test_use_tokenized_text_returns_true_for_token_lists():
    df = pd.DataFrame({
        "lyrics": [
            ["love", "forever"],
            ["dream", "ocean"]
        ]
    })

    analyzer = FeaturesAnalyzer(df, text_column="lyrics")

    assert analyzer._use_tokenized_text() is True


def test_use_tokenized_text_returns_false_for_empty_dataframe():
    df = pd.DataFrame({
        "lyrics": []
    })

    analyzer = FeaturesAnalyzer(df, text_column="lyrics")

    assert analyzer._use_tokenized_text() is False


def test_use_tokenized_text_returns_false_when_all_lyrics_are_null():
    df = pd.DataFrame({
        "lyrics": [None, np.nan]
    })

    analyzer = FeaturesAnalyzer(df, text_column="lyrics")

    assert analyzer._use_tokenized_text() is False


def test_calculate_word_count_for_string_lyrics(sample_df):
    features = FeaturesAnalyzer(sample_df, text_column="lyrics")

    features.calculate_word_count()

    assert features.df["word_count"].tolist() == [5, 4, 3]


def test_calculate_word_count_for_tokenized_lyrics(tokenized_sample_df):
    features = FeaturesAnalyzer(tokenized_sample_df, text_column="lyrics")

    features.calculate_word_count()

    assert features.df["word_count"].tolist() == [5, 4, 3]


def test_calculate_word_count_missing_lyrics_returns_zero():
    df = pd.DataFrame({
        "lyrics": [
            "love forever",
            None,
            np.nan
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_word_count()

    assert features.df["word_count"].tolist() == [2, 0, 0]


def test_calculate_word_count_empty_dataframe():
    df = pd.DataFrame(columns=["lyrics"])

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_word_count()

    assert "word_count" in features.df.columns
    assert features.df.empty


def test_calculate_unique_words_for_string_lyrics(sample_df):
    features = FeaturesAnalyzer(sample_df, text_column="lyrics")

    features.calculate_unique_words()

    assert features.df["unique_words"].tolist() == [5, 4, 3]


def test_calculate_unique_words_for_tokenized_lyrics(tokenized_sample_df):
    features = FeaturesAnalyzer(tokenized_sample_df, text_column="lyrics")

    features.calculate_unique_words()

    assert features.df["unique_words"].tolist() == [5, 4, 3]


def test_calculate_unique_words_missing_lyrics_returns_zero():
    df = pd.DataFrame({
        "lyrics": [
            "love love",
            None,
            np.nan
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_unique_words()

    assert features.df["unique_words"].tolist() == [1, 0, 0]


def test_calculate_unique_words_empty_dataframe():
    df = pd.DataFrame(columns=["lyrics"])

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_unique_words()

    assert "unique_words" in features.df.columns
    assert features.df.empty


def test_calculate_syllable_count_for_string_lyrics(sample_df):
    features = FeaturesAnalyzer(sample_df, text_column="lyrics")

    features.calculate_syllable_count()

    assert features.df["syllable_count"].tolist() == [7, 7, 3]


def test_calculate_syllable_count_for_tokenized_lyrics(tokenized_sample_df):
    features = FeaturesAnalyzer(tokenized_sample_df, text_column="lyrics")

    features.calculate_syllable_count()

    assert features.df["syllable_count"].tolist() == [7, 7, 3]


def test_calculate_syllable_count_missing_lyrics_returns_zero():
    df = pd.DataFrame({
        "lyrics": [
            "love love",
            None,
            np.nan
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_syllable_count()

    assert features.df["syllable_count"].tolist() == [2, 0, 0]


def test_calculate_syllable_count_empty_dataframe():
    df = pd.DataFrame(columns=["lyrics"])

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_syllable_count()

    assert "syllable_count" in features.df.columns
    assert features.df.empty


def test_calculate_reading_time_returns_correct_values():
    df = pd.DataFrame({
        "word_count": [200, 400, 100]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_reading_time()

    assert features.df["reading_minutes"].tolist() == [1.0, 2.0, 0.5]


def test_calculate_line_count_for_string_lyrics():
    df = pd.DataFrame({
        "lyrics": [
            "line one\nline two\nline three",
            "line one\nline two"
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_line_count()

    assert features.df["line_count"].tolist() == [3, 2]


def test_calculate_line_count_for_tokenized_lyrics():
    df = pd.DataFrame({
        "lyrics": [
            ["line", "one"],
            ["line", "two", "three"]
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_line_count()

    assert features.df["line_count"].tolist() == [2, 3]


def test_calculate_line_count_missing_lyrics_returns_zero():
    df = pd.DataFrame({
        "lyrics": [
            "line one\nline two",
            None,
            np.nan
        ]
    })

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_line_count()

    assert features.df["line_count"].tolist() == [2, 0, 0]


def test_calculate_line_count_empty_dataframe():
    df = pd.DataFrame(columns=["lyrics"])

    features = FeaturesAnalyzer(df, text_column="lyrics")

    features.calculate_line_count()

    assert "line_count" in features.df.columns
    assert features.df.empty


def test_calculate_all_structure_and_values(sample_df) -> None:
    analyzer = FeaturesAnalyzer(sample_df, text_column="lyrics")

    result = analyzer.analyze()

    # Check structure
    assert isinstance(result, pd.DataFrame)

    # Check for correct columns added
    assert "word_count" in result.columns
    assert "unique_words" in result.columns
    assert "syllable_count" in result.columns
    assert "line_count" in result.columns
