from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textblob
from sklearn.model_selection import train_test_split

from code.sentiment_classification import split_data, xgboost_classifier, svm_classifier


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


def show_results(metrics_dict, save_path):
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


def combine_sentiment_predictions(pred1, pred2, pred3, weight1, weight2, weight3):
    """
    Combine sentiment predictions from three models using weighted sum.

    Args:
    - pred1: Sentiment prediction from the first model (int, 0 or 1).
    - pred2: Sentiment prediction from the second model (int, 0 or 1).
    - pred3: Sentiment prediction from the third model (int, 0 or 1).
    - weight1: Weight for the first model's prediction (float).
    - weight2: Weight for the second model's prediction (float).
    - weight3: Weight for the third model's prediction (float).

    Returns:
    - Combined sentiment prediction (int, 0 or 1).
    """
    if weight1 + weight2 + weight3 != 1:
        raise ValueError("Weights must sum to 1.")

    combined_pred = round((pred1 * weight1) + (pred2 * weight2) + (pred3 * weight3))
    return combined_pred

if __name__ == '__main__':
    IMDB_LEM = 'data/IMDB_lemm.csv'
    IMDB_LEX = 'data/IMDB_lex.csv'

    # path_imdb_lex = Path(IMDB_LEX)
    # if not path_imdb_lex.is_file():
    #     df = pd.read_csv(IMDB_LEM)
    #     # df = df.head()
    #     df['textblob'] = df['review'].apply(analyze_sentiment_textblob)
    #     df['wbw'] = df['review'].apply(analyze_sentiment_word_by_word)
    #     df['sent'] = df['review'].apply(analyze_sentiment_sentence)
    #
    #     df.to_csv(IMDB_LEX, index=False)
    #
    # df = pd.read_csv(IMDB_LEX)
    # metrics_dict = create_metrics_dict(df)
    # show_results(metrics_dict,  f"results/Lexical_Sentiment_Performance.png")

    # ML METHODS
    df = pd.read_csv(IMDB_LEM)
    num_features = 1000
    X_train, X_test, y_train, y_test, feature_names = split_data(df, num_features)
    _, xgb_pred = xgboost_classifier(X_train, X_test, y_train, y_test, feature_names, num_features)
    _, svm_pred = svm_classifier(X_train, X_test, y_train, y_test, num_features)

    X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment, test_size=0.2, random_state=42)
    lex_pred = round(X_test.apply(analyze_sentiment_sentence))

    # Combine predictions using specified weights
    weight_xgb = 0.4
    weight_svm = 0.4
    weight_lex = 0.2

    combined_prediction = combine_sentiment_predictions(xgb_pred, svm_pred, lex_pred, weight_xgb, weight_svm,
                                                        weight_lex)
    comb_acc = accuracy_score(y_test,combined_prediction)
    comb_prec = precision_score(y_test,combined_prediction)
    comb_recall = recall_score(y_test,combined_prediction)
    comb_f1 = f1_score(y_test,combined_prediction)

    comb_dict = {
        'Combined': {
            'accuracy': comb_acc,
            'precision': comb_prec,
            'recall': comb_recall,
            'f1_score': comb_f1
        }}

    show_results(comb_dict, f"results/Combined_Sentiment_Classification.png")

    # xgb_acc = accuracy_score(y_test,xgb_pred)
    # xgb_prec = precision_score(y_test,xgb_pred)
    # xgb_recall = recall_score(y_test,xgb_pred)
    # xgb_f1 = f1_score(y_test,xgb_pred)
    #
    # svm_acc = accuracy_score(y_test, svm_pred)
    # svm_prec = precision_score(y_test, svm_pred)
    # svm_recall = recall_score(y_test, svm_pred)
    # svm_f1 = f1_score(y_test, svm_pred)
    #
    # xgb_svm_dict = {
    #     'XGB': {
    #         'accuracy': xgb_acc,
    #         'precision': xgb_prec,
    #         'recall': xgb_recall,
    #         'f1_score': xgb_f1
    #     },
    #     'SVM': {
    #         'accuracy': svm_acc,
    #         'precision': svm_prec,
    #         'recall': svm_recall,
    #         'f1_score': svm_f1
    #     }
    # }
    #
    # show_results(xgb_svm_dict, f"results/XGB_SVM_Sentiment_Classification.png")