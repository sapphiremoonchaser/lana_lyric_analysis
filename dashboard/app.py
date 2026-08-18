import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Visualize average words per song
    st.subheader("Average Words per Song")

    words_by_album = (
        album_df[
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

    st.bar_chart(
        words_by_album.set_index("album")["avg_words_per_song"]
    )

    st.caption(
        "Average number of words per song across Lana Del Rey's albums."
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

    style_df = (
        album_df[
            ["album", "year", "avg_words_per_song"]
        ]
        .drop_duplicates()
    )

    style_df["year"] = pd.to_numeric(
        style_df["year"]
    )

    style_df = style_df.sort_values("year")

    st.line_chart(
        style_df.set_index("year")["avg_words_per_song"]
    )

    st.caption(
        "Average number of words per song for each album."
    )

    # Lexical Diversity
    st.subheader("Lexical Diversity")

    diversity_df = (
        album_df[
            ["album", "year", "lexical_diversity"]
        ]
        .drop_duplicates()
    )

    diversity_df["year"] = pd.to_numeric(
        diversity_df["year"]
    )

    st.line_chart(
        diversity_df.set_index("year")["lexical_diversity"]
    )

    st.caption(
        "Average lexical diversity for each album. "
        "Higher values indicate a greater variety of unique words."
    )

    # Readability
    st.subheader("Readability")

    readability_df = (
        album_df[
            ["album", "year", "flesch_reading_ease"]
        ]
        .drop_duplicates()
    )

    readability_df["year"] = pd.to_numeric(
        readability_df["year"]
    )

    readability_df = readability_df.sort_values("year")

    st.line_chart(
        readability_df.set_index("year")[
            "flesch_reading_ease"
        ]
    )

    st.caption(
        "Flesch Reading Ease score for each album. "
        "Higher scores generally indicate easier-to-read lyrics."
    )


elif page == "Album Comparison":

    st.title("Album Comparison")

    st.write(
        "Compare lyrical characteristics across Lana Del Rey's Albums."
    )

    metric_options = {
        "Total Words": "total_words",
        "Average Words per Song": "avg_words_per_song",
        "Vocabulary Size": "vocabulary_size",
        "Lexical Diversity": "lexical_diversity",
        "Average Word Length": "average_word_length",
        "Flesch Reading Ease": "flesch_reading_ease",
        "Flesch-Kincaid": "flesch_kincaid",
        "Gunning Fog": "gunning_fog",
        "Sentiment Polarity": "sentiment_polarity",
        "Subjectivity": "subjectivity",
        "Positive Word Ratio": "positive_word_ratio",
        "Negative Word Ratio": "negative_word_ratio",
        "Positive Emotion": "emotion_Positive",
        "Negative Emotion": "emotion_Negative",
        "Anger": "emotion_Anger",
        "Anticipation": "emotion_Anticipation",
        "Disgust": "emotion_Disgust",
        "Fear": "emotion_Fear",
        "Joy": "emotion_Joy",
        "Sadness": "emotion_Sadness",
        "Surprise": "emotion_Surprise",
        "Trust": "emotion_Trust",
    }

    selected_metric = st.selectbox(
        "Choose a metric",
        options=list(metric_options.keys())
    )

    selected_column = metric_options[selected_metric]

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








elif page == "Song Explorer":

    st.title("Song Explorer")




