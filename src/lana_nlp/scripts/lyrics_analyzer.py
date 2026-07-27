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
        Return the total number of songs in the dataset.

        Returns:
            The total number of rows (songs).
        """
        return len(self.df)


    def albums(self) -> list[str]:
        """
        Alphabetical list of albums.

        Missing albums are removed before sorting.

        Returns:
            A list of unique album names.
        """
        return sorted(self.df["album"].dropna().unique())


    def number_of_songs_by_album(self) -> pd.Series:
        """
        Count the number of songs on each album.

        Groups the dataset by album and counts how many songs belong
        to each one.

        Returns:
            A pandas Series whose index contains album names and whose
            values contain the number of songs.
        """
        return (
            self.df
            .groupby("album")       # Group by album
            .size()                 # Count songs in each group
            .sort_values(ascending=False) # Show largest albums first
        )


    def song_length_stats(self) -> dict[str, float]:
        """
        Calculate descriptive statistics for songs length.

        Uses the precomputed word counts created during initialization.

        Returns:
            A dictionary containing the mean, median, minimum,
            and maximum song lengths in words.
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
        Returns the longest songs by word count.

        Args:
            n: number of songs to return

        Returns:
            A DataFrame containing the longest songs sorted by
            descending word count.
        """

        return (
            self.df
            .sort_values(
                by=["word_count"],
                ascending=False
            )
            .head(n)
        )


    def shortest_songs(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Returns the shorted songs by word count.

        Args:
            n: number of songs to return

        Returns:
            A DataFrame containing the shorted songs sorted by
            ascending word count.
        """
        return (
            self.df
            .sort_values("word_count")
            .head(n)
        )


    def album_statistics(self) -> pd.DataFrame:
        """
        Calculated summary statistics for each album.

        Statistics include:
            - number of songs
            - average words per song
            - total words on the album

        Returns:
            A DataFrame containing one row per album.
        """
        return (
            self.df
            .groupby("album")
            .agg(
                songs=("title", "count"),           # Number of songs
                avg_words=("word_count", "mean"),   # Average song length
                total_words=("word_count", "sum")   # Total words
            )
            .sort_values(
                by="songs",
                ascending=False
            )
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
