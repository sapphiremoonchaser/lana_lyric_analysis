import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from lana_nlp.visualization.preparation import (
    prepare_words_by_album
)

song_structure_metrics = {
    "avg_words_per_song": {
        "title": "Average Words per Song",
        "y_label": "Words",
    },
    "avg_lines_per_song": {
        "title": "Average Lines per Song",
        "y_label": "Lines",
    },
    "avg_words_per_line": {
        "title": "Average Words per Line",
        "y_label": "Words per Line",
    },
    "reading_minutes": {
        "title": "Average Reading Time",
        "y_label": "Minutes",
    },
}

def average_words_over_time_scatterplot(df: pd.DataFrame) -> Figure:
    """
    Create a scatter plot to show average words over time by album.
    Args:
        df: dataframe containing album and word count

    Returns:
        plotly scatter plot with avg word over time
    """
    df = prepare_words_by_album(df)

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

def create_metrics_scatter(
    df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str
) -> Figure:
    """
    Create four scatter plots with average word count, average line count, average
    words per line, and readability score.
    """
    fig = px.scatter(
        df,
        x="year",
        y=metric,
        hover_name="album",
        hover_data={
            "year": True,
            metric: ":.2f"
        },
        title=title,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_label
    )

    return fig

