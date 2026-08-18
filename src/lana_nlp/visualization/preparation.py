import pandas as pd


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