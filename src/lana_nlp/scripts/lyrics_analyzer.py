import pandas as pd
from collections import Counter


class LyricsAnalyzer:
    """
    Analyze a dataset of song lyrics.

    This class provides methods for calculating descriptive statistics,
    searching lyrics, and measuring vocabulary usage across a collection
    of songs. Derived columns such as word_count and estimated reading
    time are calculated once during initialization.
    """
    def __init__(
        self,
        lyrics_df: pd.DataFrame
    ):
        """
        Initialize the LyricsAnalyzer object.

        Creates a copy7 of the input DataFrame so the original data is not
        modified. It also calculates derived columns that are resued
        throughout the class.

        Args:
            lyrics_df: DataFrame containing song metadata and lyrics.
        """
        self.df = lyrics_df.copy()

        # count the number of words in each song
        self.df["word_count"] = (
            self.df["lyrics"] # lyrics column
            .fillna("") # replace missing lyrics with an empty string
            .str.split() # split each lyric into a list of words
            .str.len() # Coun the number of words
        )

        # Estimate the reading time assuming an average reading speed
        # of 200 words per minute
        self.df["reading_minutes"] = self.df["word_count"] / 200


    def _words(self) -> list[str]:
        """
        Return all lyrics as a single list of lowercase words.

        This private helper method centralizes text extraction so that
        multiple analysis methods can reuse the same processing logic.

        Returns:
            A list containing every word from the song.
        """

        # Combine every song's lyrics into one long string
        text = " ".join(
            self.df["lyrics"].fillna("")
        )

        # Convert everything to lowercase and split into words.
        return text.lower().split()


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
        lengths = self.df["word_count"]

        return {
            "mean": lengths.mean(),
            "median": lengths.median(),
            "min": lengths.min(),
            "max": lengths.max()
        }


    def longest_songs(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Dataframe listing longest songs by word count.
        :param n: number of songs to return
        :return: dataframe of songs
        """

        return (
            self.df
            .sort_values(by=["word_count"], ascending=False)
            .head(n)
        )


    def shortest_songs(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Dataframe listing shortest songs by word count.
        :param n: number of songs to return
        :return: dataframe of songs
        """
        return (
            self.df
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
        return (
            self.df
            .groupby("album")
            .agg(
                songs=("title", "count"),
                avg_words=("word_count", "mean"),
                total_words=("word_count", "sum")
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
                regex=False,
                na=False
            )
        ]


    def most_common_words(
        self,
        n: int = 25
    ) -> list[tuple[str, float]]:

        words = self._words()

        return Counter(words).most_common(n)


    def vocabulary_size(self) -> int:

        words = self._words()

        return len(set(words))


    def lexical_diversity(self) -> float:

        words = self._words()

        # handle division by 0
        if not words:
            return 0.0

        return len(set(words)) / len(words)
