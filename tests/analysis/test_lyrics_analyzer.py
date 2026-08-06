# Imports
import pandas as pd

from lana_nlp.analysis.lyrics_analyzer import LyricsAnalyzer


def test_calculate_derived_columns_adds_columns(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        analyzer._calculate_derived_columns()

        assert "word_count" in analyzer.df.columns
        assert "unique_words" in analyzer.df.columns
        assert "syllable_count" in analyzer.df.columns
        assert "line_count" in analyzer.df.columns
        assert "reading_minutes" in analyzer.df.columns


