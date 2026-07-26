import pandas as pd
from collections import Counter

from nltk.corpus import words


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


    def album_statistics(self) -> pd.DataFrame:
        """
        Get album statistics:
            - song count
            - average words across songs
            - total words on the album
        :return:
        """
        temp = self.df.copy()

        # create temporary column for word count
        temp["word_count"] = (
            temp["lyrics"]
            .str.split()
            .str.len()
        )

        return (
            temp
            .groupby("album")
            .agg(
                songs=("title", "count"),
                avg_words=("word_count", "mean"),
                total_words=("word_count", "count")
            )
            .sort_values(by="songs", ascending=False)
        )


    def search(
        self,
        phrase: str
    ) -> pd.DataFrame:
        """
        Search lyrics by phrase.

        :param phrase: phrase to search for
        :return:
        """
        return self.df[
            self.df["lyrics"]
            .str.contains(
                phrase,
                case=False,
                na=False
            )
        ]


    def most_common_words(
        self,
        n=25
    ):

        text = " ".join(
            self.df["lyrics"].fillna("")
        )

        words = text.lower().split()

        return Counter(words).most_common(n)


    def add_word_counts(self) -> pd.DataFrame:

        self.df["word_count"] = (
            self.df["lyrics"]
            .str.split()
            .str.len()
        )

        return self.df


    def reading_time(self):
        """
        Estimate the reading time.

        :return: dataframe including reading time
        """
        if "word_count" not in self.df:
            self.add_word_counts()

        self.df["reading_time"] = (
            self.df["word_count"] / 200
        )

        return self.df


def vocabulary_size(self) -> int:

    text = " ".join(
        self.df["lyrics"].fillna("")
    )

    words = text.lower().split()

    return len(set(words))


