# Imports
import pandas as pd
import numpy as np

from lana_nlp.analysis.lyrics_analyzer import LyricsAnalyzer


def test_calculate_derived_columns_adds_columns(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        analyzer._calculate_derived_columns()

        assert "word_count" in analyzer.df.columns
        assert "unique_words" in analyzer.df.columns
        assert "syllable_count" in analyzer.df.columns
        assert "line_count" in analyzer.df.columns
        assert "reading_minutes" in analyzer.df.columns


def test_number_of_songs_returns_int(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        assert isinstance(analyzer.number_of_songs(), int)


def test_number_of_songs_correct_value(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        assert analyzer.number_of_songs() == 3


def test_number_of_songs_empty_dataframe(empty_df) -> None:
        analyzer = LyricsAnalyzer(empty_df, text_column="lyrics")

        assert analyzer.number_of_songs() == 0


def test_number_of_songs_missing_song() -> None:
        df = pd.DataFrame({
                "song": ["Honeymoon", None, np.nan, "", "   "],
                "lyrics": ["a", "b", "c", "d", "e"]
        })

        analyzer = LyricsAnalyzer(df, text_column="lyrics")

        assert analyzer.number_of_songs() == 1



