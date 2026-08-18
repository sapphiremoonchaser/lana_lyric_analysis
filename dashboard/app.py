import streamlit as st
import pandas as pd


# Header of the dashboard
st.title("Lana Del Rey Lyric Analysis")

# load song and album level dataframes
song_df = pd.read_csv("./data/processed/song_level_stats.csv")
album_df = pd.read_csv("./data/processed/album_level_stats.csv")

# Show song level dataframe
st.write("Song-Level Data")
st.dataframe(song_df)

# Show album level dataframe
st.write("Album-Level Data")
st.dataframe(album_df)


