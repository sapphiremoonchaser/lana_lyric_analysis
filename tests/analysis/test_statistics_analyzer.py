import pandas as pd
import pytest

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


def test_song_length_stats_structure_and_value() -> None:
    df = pd.DataFrame({
        "word_count": [100, 200, 300]
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.song_length_stats()

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "mean",
        "median",
        "std",
        "min",
        "max"
    }

    assert result["mean"] == pytest.approx(200)
    assert result["median"] == pytest.approx(200)
    assert result["std"] == pytest.approx(100)
    assert result["min"] == pytest.approx(100)
    assert result["max"] == pytest.approx(300)


def test_song_length_stats_empty_dataframe():
    df = pd.DataFrame(
        columns=["word_count"]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.song_length_stats()

    assert result == {
        "mean": 0,
        "median": 0,
        "std": 0,
        "min": 0,
        "max": 0
    }


def test_longest_songs_structure_and_sorted():
    df = pd.DataFrame({
        "song": ["Honeymoon", "Ultraviolence", "Sweet"],
        "word_count": [4, 5, 3]
    })

    analyzer = StatisticsAnalyzer(df)
    result = analyzer.longest_songs(n=2)

    assert isinstance(result, pd.DataFrame)

    assert result["song"].tolist() == [
        "Ultraviolence",
        "Honeymoon"
    ]

    assert result["word_count"].tolist() == [5, 4]


def test_longest_songs_returns_empty_dataframe_for_invalid_n(sample_df):
    analyzer = StatisticsAnalyzer(sample_df)

    result = analyzer.longest_songs(n=0)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_longest_songs_empty_dataframe():
    df = pd.DataFrame(
        columns=["song", "word_count"]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.longest_songs()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_shortest_songs_structure_and_sorted():
    df = pd.DataFrame({
        "song": ["Honeymoon", "Ultraviolence", "Sweet"],
        "word_count": [4, 5, 3]
    })

    analyzer = StatisticsAnalyzer(df)
    result = analyzer.shortest_songs(n=2)

    assert isinstance(result, pd.DataFrame)

    assert result["song"].tolist() == [
        "Sweet",
        "Honeymoon"
    ]

    assert result["word_count"].tolist() == [3, 4]


def test_shortest_songs_returns_empty_dataframe_for_invalid_n(sample_df):
    analyzer = StatisticsAnalyzer(sample_df)

    result = analyzer.shortest_songs(n=0)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_shortest_songs_empty_dataframe():
    df = pd.DataFrame(
        columns=["song", "word_count"]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.shortest_songs()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_summary_by_album_returns_expected_statistics():
    df = pd.DataFrame({
        "album": ["NFR", "NFR", "Blue Banisters"],
        "song": ["A", "B", "C"],
        "word_count": [100, 200, 300],
        "reading_minutes": [0.5, 1.0, 1.5],
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_album()

    assert isinstance(result, pd.DataFrame)

    assert result.loc["NFR", "songs"] == 2
    assert result.loc["NFR", "avg_words"] == pytest.approx(150)
    assert result.loc["NFR", "median_words"] == pytest.approx(150)
    assert result.loc["NFR", "min_words"] == 100
    assert result.loc["NFR", "max_words"] == 200
    assert result.loc["NFR", "total_words"] == 300
    assert result.loc["NFR", "avg_reading_minutes"] == pytest.approx(0.75)


def test_summary_by_album_sorts_by_song_count():
    df = pd.DataFrame({
        "album": ["A", "B", "B", "B"],
        "song": ["1", "2", "3", "4"],
        "word_count": [100, 100, 200, 300],
        "reading_minutes": [1, 1, 2, 3],
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_album()

    assert result.index.tolist() == ["B", "A"]


def test_summary_by_album_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "album",
            "song",
            "word_count",
            "reading_minutes"
        ]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_album()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_longest_album_structure_and_sorted():
    df = pd.DataFrame({
        "album": [
            "Honeymoon",
            "Honeymoon",
            "Ultraviolence"
        ],
        "song": [
            "Song A",
            "Song B",
            "Song C"
        ],
        "word_count": [
            100,
            150,
            200
        ],
        "reading_minutes": [
            0.5,
            0.75,
            1.0
        ]
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.longest_album()

    assert isinstance(result, str)
    assert result == "Honeymoon"


def test_longest_album_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "album",
            "song",
            "word_count",
            "reading_minutes"
        ]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.longest_album()

    assert result == ""


def test_average_song_length_by_album_returns_expected_values():
    df = pd.DataFrame({
        "album": ["NFR", "NFR", "Blue Banisters"],
        "song": ["A", "B", "C"],
        "word_count": [100, 200, 300],
        "reading_minutes": [0.5, 1.0, 1.5],
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.average_song_length_by_album()

    assert isinstance(result, pd.DataFrame)

    assert result["album"].tolist() == [
        "NFR",
        "Blue Banisters"
    ]

    assert result["avg_words"].tolist() == [
        pytest.approx(150),
        pytest.approx(300)
    ]


def test_average_song_length_by_album_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "album",
            "song",
            "word_count",
            "reading_minutes"
        ]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.average_song_length_by_album()

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.columns.tolist() == ["album", "avg_words"]


def test_number_of_songs_by_album_structure_values_and_sorting():
    df = pd.DataFrame({
        "album": [
            "NFR",
            "NFR",
            "NFR",
            "Blue Banisters",
            "Blue Banisters",
            "Honeymoon"
        ],
        "song": ["A", "B", "C", "D", "E", "F"]
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.number_of_songs_by_album()

    assert isinstance(result, pd.Series)

    assert result.to_dict() == {
        "NFR": 3,
        "Blue Banisters": 2,
        "Honeymoon": 1
    }


def test_number_of_songs_by_album_empty_dataframe():
    df = pd.DataFrame(columns=["album", "song"])

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.number_of_songs_by_album()

    assert isinstance(result, pd.Series)
    assert result.empty


def test_summary_by_year_returns_expected_statistics():
    df = pd.DataFrame({
        "year": [2019, 2019, 2021],
        "song": ["A", "B", "C"],
        "word_count": [100, 200, 300],
        "reading_minutes": [0.5, 1.0, 1.5],
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_year()

    assert isinstance(result, pd.DataFrame)

    assert result.loc[2019, "songs"] == 2
    assert result.loc[2019, "avg_words"] == pytest.approx(150)
    assert result.loc[2019, "median_words"] == pytest.approx(150)
    assert result.loc[2019, "min_words"] == 100
    assert result.loc[2019, "max_words"] == 200
    assert result.loc[2019, "total_words"] == 300
    assert result.loc[2019, "avg_reading_minutes"] == pytest.approx(0.75)


def test_summary_by_year_sorts_by_song_count():
    df = pd.DataFrame({
        "year": [2016, 2019, 2019, 2019],
        "song": ["1", "2", "3", "4"],
        "word_count": [100, 100, 200, 300],
        "reading_minutes": [1, 1, 2, 3],
    })

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_year()

    assert result.index.tolist() == [2016, 2019]


def test_summary_by_year_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "year",
            "song",
            "word_count",
            "reading_minutes"
        ]
    )

    analyzer = StatisticsAnalyzer(df)

    result = analyzer.summary_by_year()

    assert isinstance(result, pd.DataFrame)
    assert result.empty
