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
    assert isinstance(analyzer.df["flesch_reading_ease"], float)