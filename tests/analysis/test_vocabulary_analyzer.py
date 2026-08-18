from collections import Counter

import pandas as pd
import numpy as np
import pytest

from lana_nlp.analysis.vocabulary import VocabularyAnalyzer


def test_normalize_word_lowercases_and_removes_punctuation():
    analyzer = VocabularyAnalyzer(pd.DataFrame)

    result = analyzer._normalize_word("Hello!")

    assert result == "hello"


def test_normalize_word_preserves_apostrophes():
    analyzer = VocabularyAnalyzer(pd.DataFrame)

    result = analyzer._normalize_word("Don't")

    assert result == "don't"


def test_normalize_word_removes_symbols():
    analyzer = VocabularyAnalyzer(pd.DataFrame)

    result = analyzer._normalize_word("Hello, world!")

    assert result == "helloworld"


def test_use_tokenized_text_returns_true_for_token_lists(tokenized_sample_df):
    analyzer = VocabularyAnalyzer(tokenized_sample_df)

    assert analyzer._use_tokenized_text() is np.True_


def test_use_tokenized_text_returns_false_for_token_lists(sample_df):
    analyzer = VocabularyAnalyzer(sample_df)

    assert analyzer._use_tokenized_text() is np.False_


def test_use_tokenized_text_returns_false_for_empty_dataframe():
    df = pd.DataFrame(columns=["lyrics"])

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    assert analyzer._use_tokenized_text() is np.False_


def test_get_words_returns_lowercase_words():
    df = pd.DataFrame({
        "lyrics": [
            "Fresh Out Of Fucks",
            "VEINS IN NEON"
        ]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer._get_words()

    assert result == [
        "fresh", "out", "of", "fucks",
        "veins", "in", "neon"
    ]


def test_get_words_handles_tokenized_text():
    df = pd.DataFrame({
        "lyrics": [
            ["Fresh", "Out", "Of"],
            ["VEINS", "In", "Neon"]
        ]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer._get_words()

    assert result == [
        "fresh", "out", "of",
        "veins", "in", "neon"
    ]


def test_get_words_handles_missing_lyrics():
    df = pd.DataFrame({
        "lyrics": [
            "Fresh out",
            None,
            np.nan,
            "Fucks forever"
        ]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer._get_words()

    assert result == [
        "fresh", "out",
        "fucks", "forever"
    ]


def test_get_words_empty_dataframe_returns_empty_list(empty_df):
    analyzer = VocabularyAnalyzer(empty_df, text_column="lyrics")

    result = analyzer._get_words()

    assert result == []


def test_word_frequency_structure_and_counts():
    df = pd.DataFrame({
        "lyrics": [
            "love love forever",
            "love forever",
            "forever"
        ]
    })

    analyzer = VocabularyAnalyzer(df)

    result = analyzer.word_frequency()

    assert isinstance(result, Counter)
    assert result == Counter({
        "love": 3,
        "forever": 3
    })


def test_word_frequency_is_case_insensitive():
    df = pd.DataFrame({
        "lyrics": [
            "Love LOVE love"
        ]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer.word_frequency()

    assert result == Counter({
        "love": 3
    })


def test_word_frequency_empty_dataframe(empty_df):
    analyzer = VocabularyAnalyzer(empty_df)

    result = analyzer.word_frequency()

    assert isinstance(result, Counter)
    assert result == Counter()


def test_top_n_words_structure_and_counts(sample_df):
    analyzer = VocabularyAnalyzer(sample_df)

    result = analyzer.top_n_words(n=1)

    assert result == [
        ("forever", 2)
    ]


def test_top_n_words_empty_for_n_equals_zero(sample_df):
    analyzer = VocabularyAnalyzer(sample_df)

    result = analyzer.top_n_words(n=0)

    assert result == []


def test_unique_word_count_returns_correct_count(sample_df):
    analyzer = VocabularyAnalyzer(sample_df)

    result = analyzer.unique_word_count()

    assert result == 11


def test_unique_word_count_is_case_insensitive():
    df = pd.DataFrame({
        "lyrics": ["Love love LOVE forever"]
    })

    analyzer = VocabularyAnalyzer(df)

    result = analyzer.unique_word_count()

    assert result == 2


def test_unique_word_count_empty_dataframe(empty_df):
    analyzer = VocabularyAnalyzer(empty_df)

    result = analyzer.unique_word_count()

    assert result == 0


def test_lexical_diversity_returns_correct_ratio(sample_df):
    analyzer = VocabularyAnalyzer(sample_df)

    result = analyzer.lexical_diversity()

    assert result == pytest.approx(11 / 12)


def test_lexical_diversity_returns_one_when_all_words_unique():
    df = pd.DataFrame({
        "lyrics": [
            "love forever dream ocean"
        ]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer.lexical_diversity()

    assert result == 1.0


def test_lexical_diversity_empty_dataframe(empty_df):
    analyzer = VocabularyAnalyzer(empty_df)

    result = analyzer.lexical_diversity()

    assert result == 0.0


def test_average_word_length_returns_correct_average():
    df = pd.DataFrame({
        "lyrics": ["love dream forever"]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer.average_word_length()

    # love = 4, dream = 5, forever = 7
    # 16 / 3 = 5.333...
    assert result == pytest.approx(16 / 3)


def test_average_word_length_returns_word_length_when_all_words_same_length():
    df = pd.DataFrame({
        "lyrics": ["love hope wish"]
    })

    analyzer = VocabularyAnalyzer(df, text_column="lyrics")

    result = analyzer.average_word_length()

    assert result == 4.0


def test_average_word_length_empty_dataframe(empty_df):
    analyzer = VocabularyAnalyzer(empty_df)

    result = analyzer.average_word_length()

    assert result == 0.0


# def test_song_unique_word_count_structure_and_value(sample_df):
#     analyzer = VocabularyAnalyzer(sample_df)
#
#     result = analyzer.song_unique_word_count(sample_df["lyrics"])
#
#     assert isinstance(result, int)