import nltk
import pandas as pd
import pytest

def pytest_configure():
    nltk.download(
        "opinion_lexicon",
        quiet=True
    )


@pytest.fixture
def sample_df():
    """
    Sample dataframe.
    Returns:

    """
    return pd.DataFrame(
        {
            "song": [
                "Venice Bitch",
                "Fuck it I love you",
                "Wildflower Wildfire",
            ],
            "album": [
                "NFR",
                "NFR",
                "Blue Banisters",
            ],
            "year": [
                2019,
                2019,
                2021,
            ],
            "lyrics": [
                "fresh out of fucks forever",
                "veins in neon forever",
                "star drip iv's",
            ],
        }
    )


@pytest.fixture
def tokenized_sample_df():
    """
    Sample dataframe.
    Returns:

    """
    return pd.DataFrame(
        {
            "song": [
                "Venice Bitch",
                "Fuck it I love you",
                "Wildflower Wildfire",
            ],
            "album": [
                "NFR",
                "NFR",
                "Blue Banisters",
            ],
            "year": [
                2019,
                2019,
                2021,
            ],
            "lyrics": [
                ["fresh", "out", "of", "fucks", "forever"],
                ["veins", "in", "neon", "forever"],
                ["star", "drip", "iv's"]
            ],
        }
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame(
        columns=["song", "album", "year", "lyrics"]
    )


@pytest.fixture
def single_song_df():
    return pd.DataFrame({
        "song": ["Honeymoon"],
        "album": ["Honeymoon"],
        "year": [2015],
        "lyrics": ["we make the rules"]
    })
