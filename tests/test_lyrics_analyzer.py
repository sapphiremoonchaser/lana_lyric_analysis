from collections import Counter

import numpy as np
import pytest
import pandas as pd

from src.lana_nlp.scripts.lyrics_analyzer import LyricsAnalyzer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "song": [
            "Venice Bitch",
            "Fuck it I love you",
            "Wildflower Wildfire",
        ],
        "album": [
            "NFR",
            "NFR",
            "Blue Banisters",
        ],
        "year": [
            2019,
            2019,
            2021,
        ],
        "lyrics": [
            ["fresh", "out", "of", "fucks", "forever"],
            ["veins", "in", "neon", "forever"],
            ["star", "drip", "iv's"],
        ]
    })


# ======================================================
# Initialization
# ======================================================

def test_calculate_derived_columns_empty_lyrics():
    # Create a dataframe with empty lyrics
    df = pd.DataFrame({
        "lyrics": ["", ""]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Check that all derived column names are present
    assert "word_count" in analyzer.df.columns
    assert "unique_words" in analyzer.df.columns
    assert "reading_minutes" in analyzer.df.columns

    # Check values for each derived column
    assert analyzer.df["word_count"].tolist() == [0, 0]
    assert analyzer.df["unique_words"].tolist() == [0, 0]
    assert analyzer.df["reading_minutes"].tolist() == [0, 0]


def test_calculate_derived_columns_nan_lyrics():
    df = pd.DataFrame({
        "lyrics": [None, np.nan]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Check values for each derived column
    assert analyzer.df["word_count"].tolist() == [0, 0]
    assert analyzer.df["unique_words"].tolist() == [0, 0]
    assert analyzer.df["reading_minutes"].tolist() == [0, 0]


def test_calculate_derived_columns_string_lyrics():
    df = pd.DataFrame({
        "lyrics": [
            "ice queen",
            "venice bitch"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Check for correct derived lyrics values
    assert analyzer.df["word_count"].tolist() == [2, 2]
    assert analyzer.df["unique_words"].tolist() == [2, 2]
    assert analyzer.df["reading_minutes"].tolist() == [2 / 200, 2 / 200]


def test_calculate_derived_columns_token_lyrics():
    df = pd.DataFrame({
        "lyrics": [
            ["ice", "queen"],
            ["venice", "bitch"]
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Check for correct derived lyrics values
    assert analyzer.df["word_count"].tolist() == [2, 2]
    assert analyzer.df["unique_words"].tolist() == [2, 2]
    # assert analyzer.df["reading_minutes"].tolist() == [2 / 200, 2 / 200]


def test_calculate_derived_columns_missing_values():
    df = pd.DataFrame({
        "song": [None, "Blue Jeans"],
        "album": ["Born To Die", None],
        "lyrics": ["hello world", "blue jeans"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Test derived column values
    assert analyzer.df["word_count"].tolist() == [2, 2]
    assert analyzer.df["unique_words"].tolist() == [2, 2]
    assert analyzer.df["reading_minutes"].tolist() == [2 / 200, 2 / 200]


def test_calculate_derived_columns_overwrites_existing_columns():
    df = pd.DataFrame({
        "lyrics": ["hello world"],
        "word_count": [999],
        "reading_minutes": [999]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.df["word_count"].iloc[0] == 2
    assert analyzer.df["unique_words"].iloc[0] == 2
    assert analyzer.df["reading_minutes"].iloc[0] == 2 / 200


def test_calculate_derived_columns_empty_dataframe():
    df = pd.DataFrame({"lyrics": []})

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    # Check that all derived column names are present
    assert "word_count" in analyzer.df.columns
    assert "unique_words" in analyzer.df.columns
    assert "reading_minutes" in analyzer.df.columns

    # Check that the dataframe remained empty
    assert analyzer.df.shape[0] == 0


# ======================================================
# Basic Statistics
# ======================================================

def test_number_of_songs(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    # Check number of songs value returned
    assert analyzer.number_of_songs() == 3


def test_number_of_songs_empty_dataframe():
    df = pd.DataFrame({"lyrics": []})

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.number_of_songs() == 0


def test_albums_returns_list(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.albums()

    assert isinstance(result, list)


def test_albums_returns_unique_values(sample_df):
    """
    If an album appears in the initial dataframe more than once it should only be in
    the albums list once. This test also checks that the length of the list of albums is
    correct and that the correct album names appear and are sorted.
    """
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.albums()

    assert len(result) == 2
    assert result == ["Blue Banisters", "NFR"]


def test_albums_removes_missing_values():
    """
    This tests that albums() removes None and NaN datatypes and the length of the list
    of albums
    """
    df = pd.DataFrame({
        "album": [
            "Blue Banisters",
            np.nan,
            None
        ],
        "lyrics": [
            "blue banisters",
            "coney island",
            "magical trance potion"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.albums()

    assert None not in result
    assert np.nan not in result
    assert len(result) == 1


def test_albums_empty_dataframe():
    df = pd.DataFrame({
        "album": [],
        "lyrics": []
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.albums() == []


def test_number_of_songs_by_album_returns_series(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.number_of_songs_by_album()

    assert isinstance(result, pd.Series)


def test_number_of_songs_by_album_returns_unique_albums(sample_df):
    """
    If an album appears in the initial dataframe more than once it should only be in
    the albums index once. This also checks that the Series has the correct lengths
    and is sorted by albums with the most to least songs..
    """
    expected = pd.Series(
        {
            "NFR": 2,
            "Blue Banisters": 1
        },
        name=None
    )

    expected.index.name = "album"

    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.number_of_songs_by_album()

    pd.testing.assert_series_equal(result, expected)


def test_number_of_songs_by_album_removes_missing_values():
    """
    This tests that number_of_songs_by_album() removes None and NaN datatypes. It
    also checks that the size of the series is as expected.
    """
    df = pd.DataFrame({
        "album": [
            "Blue Banisters",
            np.nan,
            None
        ],
        "lyrics": [
            "blue banisters",
            "coney island",
            "magical trance potion"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.number_of_songs_by_album()

    assert None not in result.index
    assert np.nan not in result.index
    assert result.size == 1


def test_number_of_songs_by_album_empty_dataframe():
    df = pd.DataFrame({
        "album": [],
        "lyrics": []
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.number_of_songs_by_album().size == 0


def test_songs_by_album_returns_matching_songs(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("NFR")

    assert len(result) == 2
    assert result["song"].tolist() == [
        "Venice Bitch",
        "Fuck it I love you"
    ]


def test_songs_by_album_case_insensitive(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("nfr")

    assert len(result) == 2


def test_songs_by_album_missing_album(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("Lasso")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_songs_by_album_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("Born To Die")

    assert result.empty


def test_songs_by_album_missing_album_values():
    df = pd.DataFrame({
        "album": ["Ultraviolence", None, np.nan],
        "song": ["Brooklyn Baby", "Disco", "Heavy Hitter"],
        "lyrics": [
            "beat poetry on amphetamines",
            "prostitute stare",
            "magical trance potion"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("Ultraviolence")

    assert result["album"].notna().all()


def test_songs_by_album_preserves_key_columns(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("NFR")

    assert "album" in result.columns
    assert "song" in result.columns


def test_songs_by_year_returns_matching_songs(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_year(2019)

    assert len(result) == 2
    assert result["song"].tolist() == [
        "Venice Bitch",
        "Fuck it I love you"
    ]


def test_songs_by_year_missing_year(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_year(1949)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_songs_by_year_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics", "year"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_album("Honeymoon")

    assert result.empty


def test_songs_by_year_missing_album_values():
    df = pd.DataFrame({
        "album": ["Ultraviolence", None, np.nan],
        "song": ["Brooklyn Baby", "Disco", "Heavy Hitter"],
        "lyrics": [
            "beat poetry on amphetamines",
            "prostitute stare",
            "magical trance potion"
        ],
        "year": [2014, 2009, 2010]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_year(2014)

    assert result["year"].notna().all()


def test_songs_by_year_preserves_key_columns(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.songs_by_year(2019)

    assert "song" in result.columns
    assert "year" in result.columns


def test_longest_album_returns_album_with_most_words(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.longest_album()

    assert result == 'NFR'


def test_longest_album_single_album():
    df = pd.DataFrame({
        "album": ["Ultraviolence"],
        "song": ["Ultraviolence"],
        "lyrics": ["He used to call me Poison"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_album()

    assert result == 'Ultraviolence'


def test_longest_album_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_album()

    assert result == ''


def test_longest_album_missing_album_values():
    df = pd.DataFrame({
        "album": ["Ultraviolence", None, np.nan],
        "song": ["Brooklyn Baby", "Disco", "Heavy Hitter"],
        "lyrics": [
            "beat poetry on amphetamines",
            "prostitute stare",
            "magical trance potion"
        ],
        "year": [2014, 2009, 2010]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_album()

    assert result == 'Ultraviolence'


def test_longest_album_tie_returns_first_album_sorted():
    df = pd.DataFrame({
        "album": ["Ultraviolence", "Honeymoon"],
        "song": ["Florida Kilos", "Honeymoon"],
        "lyrics": [
            "Yayo",
            "blue"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_album()

    assert result == 'Honeymoon'


# ======================================================
# Song statistics
# ======================================================

def test_song_length_stats_returns_expected_structure(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.song_length_stats()

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "mean",
        "median",
        "std",
        "min",
        "max"
    }


def test_song_length_stats_returns_correct_values(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.song_length_stats()

    assert result["mean"] == 4
    assert result["median"] == 4
    assert result["std"] == 1
    assert result["min"] == 3
    assert result["max"] == 5


def test_song_length_stats_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.song_length_stats()

    assert result == {
        "mean": 0,
        "median": 0,
        "std": 0,
        "min": 0,
        "max": 0
    }


def test_song_length_stats_single_song():
    df = pd.DataFrame({
        "album": ["Lust for Life"],
        "song": ["Cherry"],
        "lyrics": ["and all my peaches are gone"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.song_length_stats()

    assert result == {
        "mean": 6,
        "median": 6,
        "std": 0,
        "min": 6,
        "max": 6
    }


def test_longest_songs_returns_dataframe(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs(2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result["song"].tolist() == [
        "Venice Bitch",
        "Fuck it I love you"
    ]


def test_longest_songs_sorts_correctly():
    df = pd.DataFrame({
        "album": ["AKA", "Ocean Blvd"],
        "song": ["Mermaid Motel", "Taco Truck"],
        "lyrics": [
            "You call me lavender, you call me sunshine",
            "Get high, drop acid, never die, not tonight, lake placid!"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs()

    assert result["word_count"].is_monotonic_decreasing


def test_longest_songs_preserves_key_columns(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs()

    assert "album" in result.columns
    assert "song" in result.columns
    assert "lyrics" in result.columns
    assert "word_count" in result.columns


def test_longest_songs_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_longest_songs_n_equals_zero(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs(0)

    assert result.empty


def test_longest_songs_n_greater_than_dataset(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.longest_songs(5)

    assert len(result) == len(sample_df)


def test_shortest_songs_returns_dataframe(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs(2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result["song"].tolist() == [
        "Wildflower Wildfire",
        "Fuck it I love you"
    ]


def test_shortest_songs_sorts_correctly():
    df = pd.DataFrame({
        "album": ["AKA", "Ocean Blvd"],
        "song": ["Mermaid Motel", "Taco Truck"],
        "lyrics": [
            "You call me lavender, you call me sunshine",
            "caribbean blue"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs()

    assert result["word_count"].is_monotonic_increasing


def test_shortest_songs_preserves_key_columns(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs()

    assert "album" in result.columns
    assert "song" in result.columns
    assert "lyrics" in result.columns
    assert "word_count" in result.columns


def test_shortest_songs_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_shortest_songs_n_equals_zero(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs(0)

    assert result.empty


def test_shortest_songs_n_greater_than_dataset(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.shortest_songs(5)

    assert len(result) == len(sample_df)


def test_average_song_length_by_album(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.average_song_length_by_album()

    assert isinstance(result, pd.DataFrame)
    assert "album" in result.columns
    assert "avg_words" in result.columns
    assert len(result) == 2
    assert result["avg_words"].tolist() == [4.5, 3]


def test_average_song_length_by_album_empty():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.average_song_length_by_album()

    assert result.empty



# ======================================================
# Summaries
# ======================================================

def test_album_summary_returns_correct_structure(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.album_summary()

    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == [
        "songs",
        "avg_words",
        "median_words",
        "min_words",
        "max_words",
        "total_words",
        "avg_reading_minutes"
    ]
    assert result.index.name == "album"
    assert len(result) == 2

    # Test for correct stats
    assert result.loc["NFR", "songs"] == 2
    assert result.loc["NFR", "avg_words"] == 4.5
    assert result.loc["NFR", "total_words"] == 9

    # Test sorting behavior
    assert result["songs"].is_monotonic_decreasing


def test_album_summary_empty():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.album_summary()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_yearly_summary_returns_correct_structure(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.summary_by_year()

    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == [
        "songs",
        "avg_words",
        "median_words",
        "min_words",
        "max_words",
        "total_words",
        "avg_reading_minutes"
    ]
    assert result.index.name == "year"
    assert len(result) == 2

    # Test for correct stats
    assert result.loc[2019, "songs"] == 2
    assert result.loc[2019, "avg_words"] == 4.5
    assert result.loc[2019, "total_words"] == 9

    # Test sorting behavior
    assert result["songs"].is_monotonic_decreasing


def test_yearly_summary_empty():
    df = pd.DataFrame(
        columns=["album", "song", "year", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.summary_by_year()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ======================================================
# Search
# ======================================================

def test_search_found_result_structure(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.search("fucks")

    assert isinstance(result, pd.DataFrame)         # DataFrame returned
    assert len(result) == 1                         # one result based on mock data
    assert result.iloc[0]["song"] == "Venice Bitch" # correct song returned



def test_search_not_found(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.search("pineapple")

    assert result.empty


def test_search_case_insensitive(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result_1 = analyzer.search("fucks")
    result_2 = analyzer.search("FUCKS")

    assert result_1.equals(result_2)


def test_search_multiple_matches():
    df = pd.DataFrame({
        "album": ["Lust for Life", "Born To Die"],
        "song": ["Love", "Born To Die"],
        "lyrics": [
            "because we're young and in love",
            "sometimes love is not enough"
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.search("love")

    assert len(result) == 2
    assert set(result["song"]) == {
        "Love",
        "Born To Die"
    }


def test_search_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.search("love")

    assert result.empty


# ======================================================
# Vocabulary
# ======================================================

def test_top_n_words_result_structure(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.top_n_words(n=2)

    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert len(result) == 2


def test_top_n_words_more_than_available(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.top_n_words(n=10)

    assert len(result) == 3


def test_top_n_words_n_equals_zero(sample_df):
    analyzer = LyricsAnalyzer(
        sample_df,
        text_column="lyrics"
    )

    result = analyzer.top_n_words(n=0)

    assert len(result) == 0


def test_top_n_words_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.top_n_words(n=2)

    assert len(result) == 0


def test_average_word_length_zero_words():
    df = pd.DataFrame({
        "lyrics": []
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.average_word_length() == 0


def test_average_word_length_one_word():
    df = pd.DataFrame({
        "lyrics": ["forever"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.average_word_length() == 7


def test_average_word_length_multiple_words():
    df = pd.DataFrame({
        "lyrics": [
            ["cat", "house", "sun"]
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.average_word_length() == pytest.approx(11 / 3)


def test_word_frequency_structure():
    df = pd.DataFrame({
        "lyrics": [
            ["cat", "dog"],
            ["cat", "bird"]
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.word_frequency()

    assert isinstance(result, Counter)
    assert result == Counter({
        "cat": 2,
        "dog": 1,
        "bird": 1
    })


def test_word_frequency_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.word_frequency()

    assert result == Counter()


def test_vocabulary_size_structure():
    df = pd.DataFrame({
        "lyrics": [
            ["cat", "dog"],
            ["cat", "bird"],
        ]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.vocabulary_size()

    assert isinstance(result, int)
    assert analyzer.vocabulary_size() == 3


def test_vocabulary_size_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.vocabulary_size()

    assert result == 0


def test_lexical_diversity_empty_dataframe():
    df = pd.DataFrame(
        columns=["album", "song", "lyrics"]
    )

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    result = analyzer.lexical_diversity()

    assert result == 0


def test_lexical_diversity_all_unique_words():
    df = pd.DataFrame({
        "lyrics": ["cat", "dog", "bird"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.lexical_diversity() == 1.0


def test_lexical_diversity_all_same_word():
    df = pd.DataFrame({
        "lyrics": ["love", "love", "love"]
    })

    analyzer = LyricsAnalyzer(
        df,
        text_column="lyrics"
    )

    assert analyzer.lexical_diversity() == pytest.approx(1 / 3)


