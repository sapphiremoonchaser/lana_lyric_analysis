"""Run the pipeline and create the song and album level dataframes."""

from lana_nlp.pipeline.pipeline import pipeline

df = pipeline("../../../data/raw/lyrics.csv")
