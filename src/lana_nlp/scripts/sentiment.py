import pandas as pd


class LyricsAnalyzer:

    def __init__(
        self,
        lyrics_df: pd.DataFrame
    ):
        self.df = lyrics_df.copy()


    def number_of_songs(self):
        return len(self.df)


    def albums(self):
        return sorted(self.df["album"].dropna().unique())


    def songs_by_album(self):
        return (
            self.df
            .groupby("album")
            .size()
            .sort_values(ascending=False)
        )


