import pandas as pd


class LyricsAnalyzer:

    def __init__(
        self,
        lyrics_df: pd.DataFrame
    ):
        self.df = lyrics_df.copy()


    def number_of_songs(self) -> int:
        """
        Total number of songs
        :return:
        """
        return len(self.df)


    def albums(self) -> list[str]:
        """
        Alphabetical list of albums.
        :return: list of albums
        """
        return sorted(self.df["album"].dropna().unique())


    def number_of_songs_by_album(self) -> pd.Series:
        """
        Series where the index is the album name and the value is the
        number of songs on the album.
        :return: Series with song counts by album
        """
        return (
            self.df
            .groupby("album")
            .size()
            .sort_values(ascending=False)
        )


    def song_length_stats(self) -> dict[str, float]:
        """
        Dictionary with the mean, median, min, and max word counts.
        :return: dict with song length stats
        """
        lengths = self.df["lyrics"].str.split().str.len()

        return {
            "mean": lengths.mean(),
            "median": lengths.median(),
            "min": lengths.min(),
            "max": lengths.max()
        }


    def longest_songs(self, n=10) -> pd.DataFrame:
        """
        Dataframe listing longest songs by word count.
        :param n: number of songs to return
        :return: dataframe of songs
        """
        temp = self.df.copy()

        temp["word_count"] = (
            temp["lyrics"]
            .str.split()
            .str.len()
        )

        return (
            temp
            .sort_values(by=["word_count"], ascending=False)
            .head(n)
        )


    def shortest_songs(self, n=10) -> pd.DataFrame:
        """
        Dataframe listing shortest songs by word count.
        :param n: number of songs to return
        :return: dataframe of songs
        """
        temp = self.df.copy()

        temp["word_count"] = (
            temp["lyrics"]
            .str.split()
            .str.len()
        )

        return (
            temp
            .sort_values("word_count")
            .head(n)
        )