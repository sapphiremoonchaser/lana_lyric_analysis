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


def test_longest_song_sorts_correctly():
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
