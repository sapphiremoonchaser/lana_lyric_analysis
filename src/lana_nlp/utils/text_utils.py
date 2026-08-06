"""
This file contains utility functions for text columns.

Convert between string lyrics and token lyrics. Tokens are lists of strings where
each string is one word from the song.
"""


def to_text(
    text
) -> str:
    """
    If lyrics are tokens turn them into a string.
    """

    if isinstance(text, list):
        return " ".join(text)

    if isinstance(text, str):
        return text

    return ""


def to_tokens(
    text: str | list
) -> list[str]:
    """
    If lyrics are a string turn them to tokens.
    """
    if isinstance(text, list):
        return text

    if isinstance(text, str):
        return text.lower().split()

    return []