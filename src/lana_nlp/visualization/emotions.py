import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


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

