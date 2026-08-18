import pandas as pd


song_structure_metrics = {
    "avg_words_per_song": {
        "title": "Average Words per Song",
        "y_label": "Words",
        "color": "#636EFA",
    },
    "avg_lines_per_song": {
        "title": "Average Lines per Song",
        "y_label": "Lines",
        "color": "#636EFA",
    },
    "avg_words_per_line": {
        "title": "Average Words per Line",
        "y_label": "Words per Line",
        "color": "#636EFA",
    },
    "reading_minutes": {
        "title": "Average Reading Time",
        "y_label": "Minutes",
        "color": "#636EFA",
    },
}


def prepare_words_by_album(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe with words by album to be used in the visualization
    of word count over time or word count comparisons on the album level.

    Args:
        df: DataFrame to be aggregated

    Returns:
        Aggregated dataframe with word count by album sorted by year.
    """
    words_by_album = (
        df[
            ["album", "year", "avg_words_per_song"]
        ]
        .drop_duplicates()
        .sort_values("year")
    )

    words_by_album["album"] = pd.Categorical(
        words_by_album["album"],
        categories=words_by_album["album"],
        ordered=True
    )

    return words_by_album


def prepare_structure_metric_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe with words by album, lines per song, average words per line,
    average reading minutes by album.
    """
    df = (
        df.groupby(["album", "year"], as_index=False)
        .agg(
            avg_words_per_song=("word_count", "mean"),
            avg_lines_per_song=("line_count", "mean"),
            avg_words_per_line=("words_per_line", "mean"),
            avg_reading_minutes=("reading_minutes", "mean"),
        )
        .sort_values("year")
    )

    return df