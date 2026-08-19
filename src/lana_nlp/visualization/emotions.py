import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.graph_objects as go


def album_emotion_heatmap(df: pd.DataFrame) -> Figure:
    """
    Create a heatmap showing average emotion per album.
    """

    # Define emotion order
    emotion_columns = [
        "emotion_Positive",
        "emotion_Negative",
        "emotion_Anger",
        "emotion_Anticipation",
        "emotion_Disgust",
        "emotion_Fear",
        "emotion_Joy",
        "emotion_Sadness",
        "emotion_Surprise",
        "emotion_Trust",
    ]

    emotion_labels = [
        column.removeprefix("emotion_")
        for column in emotion_columns
    ]

    heatmap_df = df[["album"] + emotion_columns].copy()

    heatmap_df = heatmap_df.set_index("album")

    heatmap_df.columns = emotion_labels

    fig = px.imshow(
        heatmap_df,
        labels={
            "x": "Emotion",
            "y": "Album",
            "color": "Emotion Score"
        },
        title="Emotional Profile by Album",
        aspect="auto"
    )

    return fig


def create_emotion_heatmap(comparison_df) -> Figure:
    """
    Prepare a dataframe for emotion scores heatmap for Album Comparison.
    """
    emotion_columns = [
        "emotion_Positive",
        "emotion_Negative",
        "emotion_Anger",
        "emotion_Anticipation",
        "emotion_Disgust",
        "emotion_Fear",
        "emotion_Joy",
        "emotion_Sadness",
        "emotion_Surprise",
        "emotion_Trust",
    ]

    emotion_labels = [
        column.removeprefix("emotion_")
        for column in emotion_columns
    ]

    heatmap_data = comparison_df.set_index("album")[emotion_columns].T

    heatmap_data.index = emotion_labels

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text=heatmap_data.values,
            texttemplate="%{text:.2f}",
            colorscale="Viridis",
            colorbar_title="Score"
        )
    )

    fig.update_layout(
        title="Emotional Profile",
        xaxis_title="Album",
        yaxis_title="Emotion"
    )

    return fig


def create_emotion_bar_chart(emotion_data):
    fig = px.bar(
        emotion_data,
        x="score",
        y="emotion",
        orientation="h",
        title="Emotion Profile",
    )

    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="",
        yaxis=dict(
            categoryorder="array",
            categoryarray=emotion_data["emotion"].tolist(),
        ),
    )

    return fig