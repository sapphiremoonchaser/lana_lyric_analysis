
import streamlit as st
import pandas as pd
from lana_nlp.visualization.trends import (
    average_words_over_time_scatterplot,
    create_metrics_scatter
)

from lana_nlp.visualization.emotions import (
    album_emotion_heatmap
)

st.set_page_config(
    page_title="Lana Del Rey Lyric Analysis",
    page_icon="🍒",
    layout="wide"
)

# --------------------
# Load data
# --------------------
song_df = pd.read_csv("./data/processed/song_level_stats.csv")
album_df = pd.read_csv("./data/processed/album_level_stats.csv")

# --------------------
# Sidebar
# --------------------

st.sidebar.title("Lana Del Rey")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Lyrical Style",
        "Album Comparison",
        "Song Explorer"
    ]
)

# --------------------
# Page content
# --------------------

if page == "Overview":
    st.title("Lana Del Rey Lyric Analysis")

    st.write(
        "An exploration of Lana Del Rey's lyrics "
        "using natural language processing."
    )

    # KPI Cards
    album_count = album_df["album"].nunique()
    song_count = song_df["song"].nunique()
    first_year = album_df["year"].min()
    last_year = album_df["year"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Albums", album_count)

    with col2:
        st.metric("Songs", song_count)

    with col3:
        st.metric("First Album", first_year)

    with col4:
        st.metric("Latest Album", last_year)

    # Scatter Plot
    fig = average_words_over_time_scatterplot(album_df)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Average number of words per song across Lana Del Rey's albums."
    )

    # Emotional Profile Heatmap
    fig = album_emotion_heatmap(album_df)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Higher values indicate a higher association to a particular emotion."
    )


elif page == "Lyrical Style":

    st.title("Lyrical Style Over Time")

    # st.write(album_df.columns.tolist())

    st.write(
        "How does Lana Del Rey's lyrical style change "
        "across her discography?"
    )

    # Line chart showing average words per song
    st.subheader("Average Words per Song")

    metric_groups = {
        "Song Structure": [
            "avg_words_per_song",
            "avg_lines_per_song",
            "avg_words_per_line",
            "reading_minutes",
        ],
        "Vocabulary": [
            "unique_words",
            "lexical_diversity",
            "avg_word_length",
        ],
        "Readability": [
            "flesch_reading_ease",
            "flesch_kincaid",
            "gunning_fog",
        ],
    }

    selected_group = st.selectbox(
        "Metric Group",
        metric_groups.keys()
    )

    if selected_group == "Song Structure":

        col1, col2 = st.columns(2)

        with col1:
            fig = create_metrics_scatter(
                album_df,
                "avg_words_per_song",
                "Average Words per Song",
                "Words"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write(
                "In the future this will show average line count."
            )

        col3, col4 = st.columns(2)

        with col3:
            st.write(
                "In the future this will show average words per line"
            )

        with col4:
            st.write(
                "In the future this will show reading time."
            )

    if selected_group == "Vocabulary":
        col1, col2 = st.columns(2)

        with col1:
            fig = create_metrics_scatter(
                album_df,
                "flesch_reading_ease",
                "Flesch Reading Ease",
                "score"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Smaller negative values indicate a more difficult reading level."
            )


        with col2:
            fig = create_metrics_scatter(
                album_df,
                "flesch_kincaid",
                "Flesch-Kincaid",
                "score"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Lower values indicate a more difficult reading level."
            )

        fig = create_metrics_scatter(
            album_df,
            "gunning_fog",
            "Gunning Fog",
            "score"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher values indicate a more difficult reading level."
        )

elif page == "Album Comparison":

    st.title("Album Comparison")

    st.write(
        "Compare lyrical characteristics across Lana Del Rey's Albums."
    )

    metric_groups = {
        "Lyrical Structure": {
            "Total Words": "total_words",
            "Average Words per Song": "avg_words_per_song",
            "Vocabulary Size": "vocabulary_size",
            "Lexical Diversity": "lexical_diversity",
            "Average Word Length": "average_word_length",
        },

        "Readability": {
            "Flesch Reading Ease": "flesch_reading_ease",
            "Flesch-Kincaid": "flesch_kincaid",
            "Gunning Fog": "gunning_fog",
        },

        "Sentiment": {
            "Sentiment Polarity": "sentiment_polarity",
            "Subjectivity": "subjectivity",
            "Positive Word Ratio": "positive_word_ratio",
            "Negative Word Ratio": "negative_word_ratio",
        },

        "Emotion": {
            "Positive": "emotion_Positive",
            "Negative": "emotion_Negative",
            "Anger": "emotion_Anger",
            "Anticipation": "emotion_Anticipation",
            "Disgust": "emotion_Disgust",
            "Fear": "emotion_Fear",
            "Joy": "emotion_Joy",
            "Sadness": "emotion_Sadness",
            "Surprise": "emotion_Surprise",
            "Trust": "emotion_Trust",
        },
    }

    selected_group = st.selectbox(
        "Metric Category",
        options=metric_groups.keys()
    )

    selected_metric = st.selectbox(
        "Choose a metric",
        options=list(
            metric_groups[selected_group].keys()
        )
    )

    selected_column = metric_groups[
        selected_group
    ][selected_metric]

    comparison_df = (
        album_df[
            ["album", "year", selected_column]
        ]
        .drop_duplicates()
        .sort_values("year")
    )

    comparison_df["album"] = pd.Categorical(
        comparison_df["album"],
        categories=comparison_df["album"],
        ordered=True
    )

    st.subheader(
        f"{selected_column} by Album"
    )

    st.bar_chart(
        comparison_df.set_index("album")[selected_column]
    )

    st.caption(
        f"Comparing {selected_metric.lower()} "
        "across Lana Del Rey's albums."
    )

elif page == "Song Explorer":

    st.title("Song Explorer")

    st.write(
        "Explore the lyrical, sentiment, and emotional "
        "characteristics of individual songs."
    )

    # Drop down to choose album
    albums = sorted(
        song_df["album"].dropna().unique()
    )

    selected_album = st.selectbox(
        "Choose an album.",
        albums
    )

    # Drop down to select song
    album_songs = (
        song_df[
            song_df["album"] == selected_album
        ]
        .sort_values("song")
    )

    selected_song = st.selectbox(
        "Choose a song.",
        album_songs["song"].tolist()
    )

    selected_row = album_songs[
        album_songs["song"] == selected_song
    ].iloc[0]

    st.header(selected_song)

    st.caption(
        f"{selected_album} - {selected_row['year']}"
    )

    # KPI columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Words",
            int(selected_row["word_count"])
        )

    with col2:
        st.metric(
            "Unique Words",
            int(selected_row["unique_words"])
        )

    with col3:
        st.metric(
            "Lexical Diversity",
            f"{selected_row['lexical_diversity']:.2f}"
        )

    with col4:
        st.metric(
            "Reading Time",
            f"{selected_row['reading_minutes']:.1f}"
        )

    # Sentiment Profile
    st.subheader("Sentiment Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sentiment Polarity",
            f"{selected_row['sentiment_polarity']:.2f}"
        )

    with col2:
        st.metric(
            "Subjectivity",
            f"{selected_row['subjectivity']:.2f}"
        )

    with col3:
        st.metric(
            "Positive Language",
            f"{selected_row['positive_word_ratio']:.2%}"
        )

    with col4:
        st.metric(
            "Negative Language",
            f"{selected_row['negative_word_ratio']:.2%}"
        )

    # Readability
    st.subheader("Readability")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Flesch Reading Ease",
            f"{selected_row['flesch_reading_ease']:.1f}"
        )

    with col2:
        st.metric(
            "Flesch-Kincaid",
            f"{selected_row['flesch_kincaid']:.1f}"
        )

    with col3:
        st.metric(
            "Gunning Fog",
            f"{selected_row['gunning_fog']:.1f}"
        )

    # Emotion profile
    st.subheader("Emotion Profile")

    emotion_order = [
        "Positive",
        "Negative",
        "Anger",
        "Anticipation",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Trust",
    ]

    emotion_columns = [
        f"emotion_{emotion}"
        for emotion in emotion_order
    ]

    emotion_data = (
        selected_row[emotion_columns]
        .rename(
            lambda x: x.replace("emotion_", "")
        )
    )

    st.bar_chart(
        emotion_data
    )

    # Add the lyrics
    st.subheader("Lyrics")

    st.text_area(
        "Song lyrics",
        selected_row["lyrics"],
        height=400
    )
