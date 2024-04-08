from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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


def show_results(metrics_dict):
    labels = list(metrics_dict.keys())

    # Define the metrics to include in the plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Prepare data for each metric
    metric_data = {metric: [metrics_dict[label][metric] for label in labels] for metric in metrics}

    x = range(len(labels))
    width = 0.2  # Width of each bar

    plt.figure(figsize=(15, 8))

    # Plotting bars for each metric
    for i, metric in enumerate(metrics):
        bars = plt.bar([pos + i * width for pos in x], metric_data[metric], width=width, label=metric.capitalize())

        # Adding value annotations above each bar
        for bar, value in zip(bars, metric_data[metric]):
            plt.text(bar.get_x() + bar.get_width() / 2, value, round(value, 2), ha='center', va='bottom')

    plt.xticks([pos + width for pos in x], labels)
    plt.xlabel('Methods')
    plt.ylabel('Metrics')
    plt.title('Sentiment classification results')
    plt.legend()
    plt.tight_layout()

    save_path = f"results/Lexical_Sentiment_Performance.png"
    path_save = Path(save_path)
    if not path_save.is_file():
        plt.savefig(save_path)

    plt.show()


def create_metrics_dict(df):
    accuracy_textblob = accuracy_score(df.textblob.astype(int), df.sentiment)
    accuracy_wbw = accuracy_score(df.wbw.astype(int), df.sentiment)
    accuracy_sent = accuracy_score(df.sent.astype(int), df.sentiment)

    precision_textblob = precision_score(df.textblob.astype(int), df.sentiment)
    precision_wbw = precision_score(df.wbw.astype(int), df.sentiment)
    precision_sent = precision_score(df.sent.astype(int), df.sentiment)

    recall_textblob = recall_score(df.textblob.astype(int), df.sentiment)
    recall_wbw = recall_score(df.wbw.astype(int), df.sentiment)
    recall_sent = recall_score(df.sent.astype(int), df.sentiment)

    f1_textblob = f1_score(df.textblob.astype(int), df.sentiment)
    f1_wbw = f1_score(df.wbw.astype(int), df.sentiment)
    f1_sent = f1_score(df.sent.astype(int), df.sentiment)

    metrics_dict = {
        'TextBlob': {
            'accuracy': accuracy_textblob,
            'precision': precision_textblob,
            'recall': recall_textblob,
            'f1_score': f1_textblob
        },
        'Word by Word': {
            'accuracy': accuracy_wbw,
            'precision': precision_wbw,
            'recall': recall_wbw,
            'f1_score': f1_wbw
        },
        'Sentiment Analysis': {
            'accuracy': accuracy_sent,
            'precision': precision_sent,
            'recall': recall_sent,
            'f1_score': f1_sent
        }
    }

    return metrics_dict


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

    metrics_dict = create_metrics_dict(df)
    show_results(metrics_dict)
