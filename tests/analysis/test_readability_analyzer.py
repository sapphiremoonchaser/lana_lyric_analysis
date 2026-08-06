# Imports
import pandas as pd
import numpy as np

from lana_nlp.analysis.readability import ReadabilityAnalyzer


def test_flesch_reading_ease_returns_calculated_column(sample_df) -> None:
    analyzer = ReadabilityAnalyzer(sample_df, text_column="lyrics")

    analyzer.flesch_reading_ease()

    # Test structure
    assert "flesch_reading_ease" in analyzer.df.columns

    # Test value was calculated
    assert pd.api.types.is_float_dtype(analyzer.df["flesch_reading_ease"])


def test_flesch_kincaid_returns_calculated_column(sample_df) -> None:
    analyzer = ReadabilityAnalyzer(sample_df, text_column="lyrics")

    analyzer.flesch_kincaid()

    # Test structure
    assert "flesch_kincaid" in analyzer.df.columns

    # Test value was calculated
    assert pd.api.types.is_float_dtype(analyzer.df["flesch_kincaid"])


def test_gunning_fog_returns_calculated_column(sample_df) -> None:
    analyzer = ReadabilityAnalyzer(sample_df, text_column="lyrics")

    analyzer.gunning_fog()

    # Test structure
    assert "gunning_fog" in analyzer.df.columns

    # Test value was calculated
    assert pd.api.types.is_float_dtype(analyzer.df["gunning_fog"])


def test_calculate_all_structure_and_values(sample_df) -> None:
    analyzer = ReadabilityAnalyzer(sample_df, text_column="lyrics")

    result = analyzer.analyze()

    # Check structure
    assert isinstance(result, pd.DataFrame)

    # Check for correct columns added
    assert "flesch_reading_ease" in result.columns
    assert "flesch_kincaid" in result.columns
    assert "gunning_fog" in result.columns
