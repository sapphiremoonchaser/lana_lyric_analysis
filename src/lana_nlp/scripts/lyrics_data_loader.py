"""
LyricsDataLoader does the following:
    - reads the csv
    - verifies required columns exist
    - stores the dataframe
    - returns a copy of the dataframe
"""

from pathlib import Path

import pandas as pd
from jinja2.utils import missing

REQUIRED_COLUMNS = {
    "artist",
    "album",
    "song",
    "year",
    "lyrics"
}


class LyricsDataLoader:

    def __init__(
        self,
        filepath: str | Path
    ):
        # Setup the initial variables
        self.filepath = Path(filepath)
        self.df = None

    def load(self):
        df = pd.read_csv(self.filepath)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = df
        return self.df.copy()


