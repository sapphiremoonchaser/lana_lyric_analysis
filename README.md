# Lana Del Rey Lyric Analyzer


## Overview
The Lana Del Rey Lyric Analyzer explores Lana's lyrics across her albums using natural language processing (NLP) 
and data analysis. Lyrics are preprocessed and cleaned before a variety of statistical and linguistic 
metrics are calculated. 

The project examines metrics such as word count, unique words, lexical diversity, reading time, readability, sentiment, 
and emotional profile. The results are presented through interactive visualizations and a Streamlit dashboard featuring 
an album comparison page and a song explorer.

## Research Questions
How has Lana Del Rey's lyrical structure changed over time?   
How has the diversity and complexity of her vocabulary changed over time?   
How to albums compare?

## Features
**Album Comparison** - Compare albums across lyrical and NLP metrics, including word count, vocabulary size, lexical 
diversity, readability, sentiment, and emotional profile.

**Song Explorer** - Explore individual songs and view their lyrics and NLP metrics.

**Lyrics Analysis** - Calculate word count, unique words, lexical diversity, line count, reading time, and readability 
metrics.

**Sentiment Analysis** - Analyze sentiment polarity and subjectivity using NLP techniques.

**Emotion Analysis** - Examine the presence of emotions including anger, anticipation, digust, fear, 
joy, sadness, surprise, and trust.

**Interactive Visualizations** - Explore lyrical patterns and differences through interactive charts and visualizations.

**Album-Specific Styling** - Visualizations use custom color palettes inspired by each album.

## Data
Lyrics were collectedfrom the [Lana Del Rey Wiki](https://lanadelrey.fandom.com/wiki/Lana_Del_Rey_wiki) and compiled 
into a CSV dataset. The dataset contains album, year, song, and lyrics.  
  
The lyrics were then cleaned and preprocessed for analysis and natural language processeing.

## NLP & Analysis

### Text Processing
Basic text processing is done first to normalize lyrics. This includes converting text to lowercase, removing 
annotations and excess whitespace, expanding contractions, and removing punctuation.

Additional processing prepares the lyrics for NLP analysis. This includes tokenizing the lyrics into individual words, 
removing stopwords (such as "the" and "and"), and lemmatizing words so that related forms such as "love" and "loved" 
are treated as the same word.

Two versions of the cleaned lyrics are maintained: a basic cleaned version use for general feature calculations and an 
NLP-cleaned version used for vocabulary, sentiment, emotion, and readability analysis.

### Metrics

#### Basic Lyrics Metrics

* Word Count
* Unique Words
* Line Count
* Reading Time

#### Vocabulary
* Vocabulary Size
* Lexical Diversity
* Word Length

#### Readability
* Flesch Reading Ease
* Flesch-Kincaid Grade Level
* Gunning Fog Index

#### Sentiment
* Sentiment Polarity
* Subjectivity
* Positive Word Ratio
* Negative Word Ratio

#### Emotional Profile
* NRC emotion categories: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust

## Dashboard

### Overview

![Overview](images/01_overview.png)

### Lyrical Style

![Lyrical Structure Over Time](images/02_song_structure_over_time.png)

![Sentiment Over Time](images/03_sentiment_over_time.png)

### Album Comparison

![Album Comparison](images/04_vocab_album_comparison.png)

### Song Explorer

![Song Explorer](images/05_song_explorer.png)

![Song Explorer Word Cloud](images/06_song_explorer_2.png)

## Technologies

### Programming & Data Analysis

* **Python** — Primary programming language
* **pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations

### Natural Language Processing

* **NLTK** — Tokenization, stopword removal, and lemmatization
* **TextBlob** — Sentiment analysis
* **VADER** — Sentiment analysis
* **textstat** — Readability metrics

### Visualization & Dashboard

* **Plotly** — Interactive data visualizations
* **Streamlit** — Interactive web dashboard
* **Matplotlib** — Data visualization

## Project Structure

```aiignore
lana_lyric_analysis/ 
│ 
├── data/ 
│ ├── raw/ # Original lyric dataset 
│ └── processed/ # Cleaned and processed data 
│ 
├── examples/ # Example notebooks/scripts 
│ 
├── images/ # Dashboard and visualization images 
│ 
├── src/ 
│ └── lana_nlp/ 
│   ├── dashboard/ 
│   │ ├── visualizations/ 
│   │ │ ├── color_palettes.py # Album color palettes 
│   │ │ ├── comparisons.py # Album comparison visualizations 
│   │ │ ├── emotions.py # Emotion visualizations 
│   │ │ ├── preparation.py # Visualization data preparation 
│   │ │ └── trends.py # Trends over time 
│   └── app.py # Streamlit dashboard 
│ 
│   ├── features/ 
│   │ ├── basic_features.py # Basic lyric statistics 
│   │ ├── readability.py # Readability metrics 
│   │ ├── sentiment.py # Sentiment analysis 
│   │ └── vocabulary.py # Vocabulary analysis 
│   ├── pipeline/ 
│   │ ├── build_pipeline.py # Pipeline construction 
│   │ ├── data_loader.py # Data loading 
│   │ ├── pipeline.py # Data processing pipeline 
│   │ └── text_cleaner.py # Text cleaning and preprocessing 
│ │ └── utils/ 
│       └── text_utils.py # Text utility functions 
├── tests/ # Unit tests 
├── .gitignore # Files excluded from Git 
├── pyproject.toml # Project configuration and metadata 
└── README.md # Project documentation
```

## Installation & Usage

1. Clone the repository  

`git clone https://github.com/sapphiremoonchaser/lana_lyric_analysis.git cd lana_lyric_analysis`  


2. Create a virtual environment and activate

3. Install dependencies from pyproject.toml  

`pip install -e .`

4. Save lyrics  

Create a csv with album, song, year, and lyrics and save in data/raw/lyrics.csv

5. Save the NRC emotion lexicon in data/raw/NRC.csv

source: https://www.kaggle.com/datasets/wjburns/nrc-emotion-lexicon

6. Run the Streamlit Dashboard  

`streamlit run src/lana_nlp/dashboard/app.py`


## Future Improvements
* Expand sentiment analysis and visualization
* TF-IDF to find words that describe each album
* Implement topic modeling using Latent Dirichlet Allocation (LDA)
* Add artist-to-artist comparisons


## Author
Heather Hill  
[LinkedIn](www.linkedin.com/in/heather-gwyn)  
[GitHub](https://github.com/sapphiremoonchaser)
