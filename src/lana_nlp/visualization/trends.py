import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure


def average_words_over_time(df: pd.DataFrame) -> Figure:
    """
    Create a scatter plot to show average words over time by album.
    Args:
        df: dataframe containing album and word count

    Returns:
        plotly scatter plot with avg word over time
    """
    fig = px.scatter(
        df,
        x="year",
        y="avg_words_per_song",
        hover_name="album",
        labels={
            "year": "Year",
            "avg_words_per_song": "Average Words per Song"
        },
        title="Average Words per Song over Time"
    )

    return fig