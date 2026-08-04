import nltk


def pytest_configure():
    nltk.download(
        "opinion_lexicon",
        quiet=True
    )