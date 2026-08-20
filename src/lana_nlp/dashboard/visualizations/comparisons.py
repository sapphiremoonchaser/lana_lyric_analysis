from random import random

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from wordcloud import WordCloud
import ast




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


def create_album_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str
) -> Figure:
    fig = px.bar(
        df,
        x="album",
        y=metric,
        title=title
    )

    fig.update_layout(
        xaxis_title="Album",
        yaxis_title=y_label
    )

    return fig


def create_album_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str
) -> Figure:
    fig = px.box(
        df,
        x="album",
        y=metric,
        color="album",
        points="all",
        hover_name="song",
        hover_data={
            "album": False,
            metric: ":.1f"
        },
        title=title
    )

    fig.update_layout(
        xaxis_title="Album",
        yaxis_title=y_label,
        showlegend = False
    )

    return fig


def create_wordcloud(text, palette):
    """
    Create a word cloud based on a string of text.

    Parameters:
        text (str): The string of text. or tokenized lyrics
        palette (str): matplotlib colormap name used for the word cloud.

    Returns:
        WordCloud: generated word cloud.
    """
    if isinstance(text, list):
        text = " ".join(text)

    def color_func(
            word,
            font_size,
            position,
            orientation,
            random_state=None,
            **kwargs
    ):
        return random_state.choice(palette)

    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        color_func=color_func,
        random_state=42
    ).generate(text)

    return wordcloud