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


def test_songs_by_year():
    # test datatype
    # test shape
    # test value
    pass


def test_longest_album():
    # test value
    pass


# ======================================================
# Song statistics
# ======================================================

def test_song_length_stats():
    # Test datatype
    # Test values
    pass


def test_song_length_stats_empty():
    pass


def test_longest_songs():
    # Test datatype
    # Test shape
    # test value
    pass


def test_longest_songs_ties():
    pass


def test_shortest_songs():
    # Test datatype
    # Test shape
    # test value
    pass


def test_shortest_songs_ties():
    pass


def test_average_song_length_by_album():
    # Test datatype
    # test shape
    # test value
    pass


# ======================================================
# Summaries
# ======================================================

def test_album_summary():
    # Test datatype
    # Test shape
    # Test column names
    # Test values
    pass


def test_yearly_summary():
    # Test datatype
    # Test shape
    # Test column names
    # Test values
    pass


# ======================================================
# Search
# ======================================================

def test_search_found():
    # Test datatype
    # Test sape
    # Test columns
    # Test values
    pass


def test_search_not_found():
    pass


def search_case_insensitive():
    # Test values
    pass


def test_search_multiple_matches():
    pass


def test_search_empty_query():
    pass


# ======================================================
# Vocabulary
# ======================================================

def test_top_n_words():
    # test datatype
    # test length
    # test values
    pass


def test_top_n_words_more_than_available():
    pass


def test_top_n_words_empty():
    pass


def test_average_word_length_zero_words():
    pass


def test_average_word_length_single_word():
    pass


def test_average_word_length():
    # test value
    pass


def test_word_frequency():
    # test value
    pass


def test_word_frequency_missing_word():
    pass


def test_vocabulary_size():
    # test value
    pass


def test_vocabulary_size_empty():
    pass


def test_lexical_diversity_empty_words():
    pass


def test_lexical_diversity():
    # test value
    pass


def test_lexical_diversity_all_same_word():
    pass


def test_lexical_diversity_all_unique_words():
    pass
