from src.lana_nlp.scripts.text_cleaner import TextCleaner


def test_lowercase():

    cleaner = TextCleaner()

    result = cleaner.clean_text(
        "Blue Jeans"
    )

    assert result == "blue jeans"

