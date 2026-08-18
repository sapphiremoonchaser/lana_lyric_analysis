"""
LyricsDataLoader does the following:
    - reads the csv
    - verifies required columns exist
    - stores the dataframe
    - returns a copy of the dataframe
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "artist",
    "album",
    "song",
    "year",
    "lyrics"
}


class LyricsDataLoader:
    """
    This class loads the lyrics data from a csv file.
    """
    def __init__(
        self,
        filepath: str | Path
    ):
        # Set up the initial variables
        self.filepath = Path(filepath)
        self.df = None

    def load(self):
        """
        Load the lyrics data from a csv file.
        """
        df = pd.read_csv(self.filepath)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = df
        return self.df.copy()


