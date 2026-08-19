import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def average_words_by_album(df: pd.DataFrame) -> Figure:
    """
    Create a bar chart showing average words per song for each album.
    """
    fig = px.bar(
        df,
        x="avg_words_per_song",
        y="album",
        orientation="h",
        labels={
            "avg_words_per_song": "Average Words per Song",
            "album": "Album"
        },
        title="Average Words per Song by Album"
    )

    return fig