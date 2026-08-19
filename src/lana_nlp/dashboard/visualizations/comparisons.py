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


def create_wordcloud(text, palatte):
    """
    Create a wordcloud based on a string of text.
    """
    if isinstance(text, str):
        try:
            text = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

    if isinstance(text, list):
        text = " ".join(text)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="twilight"
    ).generate(text)

    return wordcloud