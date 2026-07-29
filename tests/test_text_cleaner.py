from src.lana_nlp.scripts.text_cleaner import TextCleaner


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