import pytest
import numpy as np

from lana_nlp.pipeline.text_cleaner import TextCleaner


def test_remove_annotations_removes_verse_label():
    cleaner = TextCleaner()

    text = "[Verse 1] \nI was walking down the street"

    result = cleaner.remove_annotations(text)

    assert result == " \nI was walking down the street"


def test_remove_annotations_removes_different_annotations():
    cleaner = TextCleaner()

    text = "[Verse 1] Hello [Chorus] world [Instrumental]"

    result = cleaner.remove_annotations(text)

    assert result == " Hello  world "


def test_remove_annotations_preserves_text_without_annotations():
    cleaner = TextCleaner()

    text = "Hello, I love you"

    result = cleaner.remove_annotations(text)

    assert result == text


def test_remove_annotations_removes_empty_annotation():
    cleaner = TextCleaner()

    text = "Hello [] world"

    result = cleaner.remove_annotations(text)

    assert result == "Hello  world"


def test_lowercase():

    cleaner = TextCleaner()

    result = cleaner.clean_text(
        "Blue Jeans"
    )

    assert result == "blue jeans"


def test_removes_punctuation():

    cleaner = TextCleaner()

    result = cleaner.clean_text(
        "Hello!!!"
    )

    assert result == "hello"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "hello    world",
            "hello world"
        ),
        (
            "hello\nworld",
            "hello world"
        ),
        (
            "hello\tworld",
            "hello world"
        ),
        (
            "  hello world  ",
            "hello world"
        ),
        (
            "  hello   world\n\tagain  ",
            "hello world again"
        ),
        (
            "",
            ""
        ),
    ]
)
def test_remove_whitespace(text, expected):
    cleaner = TextCleaner()

    result = cleaner.remove_whitespace(text)

    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "I love you",
            ["I", "love", "you"]
        ),
        (
            "I love you!",
            ["I", "love", "you", "!"]
        ),
        (
            "Hello, world.",
            ["Hello", ",", "world", "."]
        ),
        (
            "love   you",
            ["love", "you"]
        ),
        (
            "",
            []
        ),
    ]
)
def test_tokenize(text, expected):
    cleaner = TextCleaner()

    result = cleaner.tokenize(text)

    assert result == expected


def test_tokenize_preserves_contractions():
    cleaner = TextCleaner()

    result = cleaner.tokenize("I don't wanna leave")

    assert result == ["I", "do", "n't", "wan", "na", "leave"]


def test_basic_clean():
    cleaner = TextCleaner()

    text = "[Verse 1]  Hello, WORLD!"

    result = cleaner.basic_clean(text)

    assert result == "hello world"


def test_basic_clean_removes_multiple_annotations():
    cleaner = TextCleaner()

    text = "[Verse 1] Hello [Chorus] Beautiful World [Verse 2]"

    result = cleaner.basic_clean(text)

    assert result == "hello beautiful world"


def test_basic_clean_handles_missing_text():
    cleaner = TextCleaner()

    result = cleaner.basic_clean(None)

    assert result == ""


def test_basic_clean_handles_nan():
    cleaner = TextCleaner()

    result = cleaner.basic_clean(np.nan)

    assert result == ""


def test_basic_clean_handles_empty_string():
    cleaner = TextCleaner()

    result = cleaner.basic_clean("")

    assert result == ""


def test_basic_clean_normalizes_lyric_whitespace():
    cleaner = TextCleaner()

    text = "[Verse 1]\nHello,\nbeautiful   world!"

    result = cleaner.basic_clean(text)

    assert result == "hello beautiful world"


