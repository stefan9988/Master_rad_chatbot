from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textblob


# Downloading NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')


def analyze_sentiment_word_by_word(text):
    sid = SentimentIntensityAnalyzer()
    tokens = word_tokenize(text)

    total_score = 0
    for word in tokens:
        total_score += sid.polarity_scores(word)['compound']

    average_score = total_score / len(tokens)

    if average_score > 0.2:
        sentiment_label = 1
    elif average_score < -0.2:
        sentiment_label = 0
    else:
        sentiment_label = 0.5

    return sentiment_label


def analyze_sentiment_sentence(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['compound'] > 0.2:
        sentiment_label = 1
    elif scores['compound'] < -0.2:
        sentiment_label = 0
    else:
        sentiment_label = 0.5

    return sentiment_label


def analyze_sentiment_textblob(text):
    score = textblob.TextBlob(text).sentiment.polarity
    if score > 0.2:
        sentiment_label = 1
    elif score < -0.2:
        sentiment_label = 0
    else:
        sentiment_label = 0.5

    return sentiment_label


def show_results(accuracy_textblob, accuracy_wbw, accuracy_sent):
    labels = ['TextBlob', 'Word by Word', 'Sentiment Analysis']

    # Heights of the bars
    heights = [accuracy_textblob, accuracy_wbw, accuracy_sent]

    # Plotting the bar plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(labels, heights)

    # Adding the value above each bar
    for bar, height in zip(bars, heights):
        plt.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2),
                 ha='center', va='bottom')

    # Adding the title and labels
    plt.title('Accuracy Comparison')
    plt.xlabel('Methods')
    plt.ylabel('Accuracy')

    save_path = f"results/Lexical_Sentiment_Classification.png"
    path_save = Path(save_path)
    if not path_save.is_file():
        plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':
    IMDB_LEM = 'data/IMDB_lemm.csv'
    IMDB_LEX = 'data/IMDB_lex.csv'

    path_imdb_lem = Path(IMDB_LEX)
    if not path_imdb_lem.is_file():
        df = pd.read_csv(IMDB_LEM)
        df = df.head()
        df['textblob'] = df['review'].apply(analyze_sentiment_textblob)
        df['wbw'] = df['review'].apply(analyze_sentiment_word_by_word)
        df['sent'] = df['review'].apply(analyze_sentiment_sentence)

        df.to_csv(IMDB_LEX, index=False)

    df = pd.read_csv(IMDB_LEX)

    accuracy_textblob = accuracy_score(df.textblob.astype(int), df.sentiment)
    accuracy_wbw = accuracy_score(df.wbw.astype(int), df.sentiment)
    accuracy_sent = accuracy_score(df.sent.astype(int), df.sentiment)

    accuracy_textblob = round(accuracy_textblob, 2)
    accuracy_wbw = round(accuracy_wbw, 2)
    accuracy_sent = round(accuracy_sent, 2)

    show_results(accuracy_textblob, accuracy_wbw, accuracy_sent)
