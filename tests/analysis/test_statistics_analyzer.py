import pandas as pd

from lana_nlp.analysis.statistics import StatisticsAnalyzer


def test_total_word_count_structure_and_value() -> None:
    # Not using sample_df bc LyricsAnalyzer is responsible for creating word_count
    # column
    df = pd.DataFrame({
        "song": [
            "Venice Bitch",
            "Fuck it I love you",
            "Wildflower Wildfire"
        ],
        "word_count": [5, 4, 3]
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.total_word_count()

    assert isinstance(result, int)
    assert result == 12


def test_total_word_count_empty_dataframe() -> None:
    df = pd.DataFrame(
        columns=["song", "word_count"]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.total_word_count()

    assert result == 0


def test_average_song_length_structure_and_value() -> None:
    # Not using sample_df bc LyricsAnalyzer is responsible for creating word_count
    # column
    df = pd.DataFrame({
        "song": [
            "Venice Bitch",
            "Fuck it I love you",
            "Wildflower Wildfire"
        ],
        "word_count": [5, 4, 3]
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.average_song_length()

    assert isinstance(result, float)
    assert result == 4


def test_average_song_length_empty_dataframe() -> None:
    df = pd.DataFrame(
        columns=["song", "word_count"]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.average_song_length()

    assert result == 0


