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


def test_number_of_songs_structure_and_value(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        assert isinstance(analyzer.number_of_songs(), int)
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


def test_albums_structure_and_value(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        assert isinstance(analyzer.albums(), list)
        assert len(analyzer.albums()) == 2
        assert analyzer.albums() == [
                "Blue Banisters",       # also tests sorting
                "NFR"
        ]


def test_albums_empty_dataframe(empty_df) -> None:
        analyzer = LyricsAnalyzer(empty_df, text_column="lyrics")

        assert analyzer.albums() == []


def test_albums_missing_album() -> None:
        df = pd.DataFrame({
                "song": ["Honeymoon", "Love", "Doin' Time", "Kill Kill", "Bad Disease"],
                "album": ["Honeymoon", None, np.nan, "", "   "],
                "lyrics": ["a", "b", "c", "d", "e"]
        })

        analyzer = LyricsAnalyzer(df, text_column="lyrics")

        assert len(analyzer.albums()) == 1


def test_songs_by_album_structure_and_value(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        songs_by_album = analyzer.songs_by_album("NFR")

        # Test structure
        assert isinstance(songs_by_album, pd.DataFrame)
        assert songs_by_album.shape[0] == 2
        assert "song" in songs_by_album.columns
        assert "album" in songs_by_album.columns

        # Test correct album is returned
        assert set(songs_by_album["album"]) == {"NFR"}
        assert set(songs_by_album["song"]) == {
                "Venice Bitch",
                "Fuck it I love you"
        }


def test_songs_by_album_empty_dataframe(empty_df) -> None:
        analyzer = LyricsAnalyzer(empty_df, text_column="lyrics")

        songs_by_album = analyzer.songs_by_album("NFR")

        assert songs_by_album.empty


def test_songs_by_year_structure_and_value(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        songs_by_year = analyzer.songs_by_year(2019)

        # Test structure
        assert isinstance(songs_by_year, pd.DataFrame)
        assert songs_by_year.shape[0] == 2
        assert "song" in songs_by_year.columns
        assert "year" in songs_by_year.columns

        # Test correct album is returned
        assert set(songs_by_year["year"]) == {2019}
        assert set(songs_by_year["song"]) == {
                "Venice Bitch",
                "Fuck it I love you"
        }


def test_songs_by_year_empty_dataframe(empty_df) -> None:
        analyzer = LyricsAnalyzer(empty_df, text_column="lyrics")

        songs_by_year = analyzer.songs_by_year(2019)

        assert songs_by_year.empty


def test_calculate_all_structure_and_values(sample_df) -> None:
        analyzer = LyricsAnalyzer(sample_df, text_column="lyrics")

        result = analyzer.analyze()

        # Check structure
        assert isinstance(result, pd.DataFrame)

        # Check for correct columns added
        assert "word_count" in result.columns
        assert "unique_words" in result.columns
        assert "syllable_count" in result.columns
        assert "line_count" in result.columns
        assert "reading_minutes" in result.columns

