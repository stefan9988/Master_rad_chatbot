import os
import pickle
import re
from typing import List, Tuple
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_stopwords_and_punctuation(question: str) -> str:
    """
        Remove stopwords and punctuation from the input question using NLTK and regular expressions.

        Parameters:
        - question (str): Input question to be processed.

        Returns:
        - str: Processed question with stopwords and punctuation removed.
        """
    word_list = nltk.word_tokenize(question.lower())
    stop = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    exclude_words = {"what", "where", "who", "why", "how","which","when","whom","whose"}
    stop -= exclude_words
    pattern = r'[^\w\s]'

    processed_words = [re.sub(pattern, '', word) for word in word_list if word not in stop]
    processed_question = ' '.join(processed_words)

    return processed_question


def lemm(question: str) -> str:
    """
    Perform lemmatization on the input question using NLTK's WordNetLemmatizer.

    Parameters:
    - question (str): Input question to be lemmatized.

    Returns:
    - str: Lemmatized output of the input question.
    """
    word_list = nltk.word_tokenize(question)
    lemmatizer = WordNetLemmatizer()

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


def stem(question: str) -> str:
    """
    Perform stemming on the input question using NLTK's PorterStemmer.

    Parameters:
    - question (str): Input question to be stemmed.

    Returns:
    - str: Stemmed output of the input question.
    """
    word_list = nltk.word_tokenize(question)
    stemmer = PorterStemmer()

    stemmed_output = ' '.join([stemmer.stem(w) for w in word_list])
    return stemmed_output


def get_stemmed_questions_series(questions: List[str]) -> pd.Series:
    """
    Apply stemming to a series of questions and return a Pandas Series.

    Parameters:
    - questions (List[str]): List of input questions to be stemmed.

    Returns:
    - pd.Series: Series of stemmed questions.
    """
    stemmed_questions = pd.Series([stem(q) for q in questions], name='Question')
    return stemmed_questions


def get_lemm_questions_series(questions: List[str]) -> pd.Series:
    """
    Apply lemmatization to a series of questions and return a Pandas Series.

    Parameters:
    - questions (List[str]): List of input questions to be lemmatized.

    Returns:
    - pd.Series: Series of lemmatized questions.
    """
    lemm_questions = pd.Series([lemm(q) for q in questions], name='Question')
    return lemm_questions


def tf_idf(train_s: pd.Series, test_s: pd.Series or None, num_features: int or None) -> Tuple:
    """
    Compute TF-IDF features for the input training and testing series using sklearn's TfidfVectorizer.

    Parameters:
    - train_s (pd.Series): Training series of questions.
    - test_s (pd.Series or None): Testing series of questions. (None if not provided)
    - num_features (int or None): Number of features to retain. (None if not specified)

    Returns:
    - Tuple: Tuple containing train vectors, test vectors, and feature names.
    """
    vectorizer = TfidfVectorizer(max_features=num_features)
    train_vectors = vectorizer.fit_transform(train_s)

    vectorizer_filename = r'data/vectorizer.pkl'
    if not os.path.exists(vectorizer_filename):
        with open(vectorizer_filename, 'wb') as file:
            pickle.dump(vectorizer, file)

    if test_s is not None:
        test_vectors = vectorizer.transform(test_s)
    else:
        test_vectors = None
    feature_names = vectorizer.get_feature_names_out()


    return train_vectors, test_vectors, feature_names

def assign_sentiment_values(df: pd.DataFrame) -> pd.DataFrame:
    """
        Assign binary values to the 'sentiment' column based on the original values.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'sentiment' column containing text data.

        Returns:
        pd.DataFrame: DataFrame with the 'sentiment' column updated to binary values.
        """
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/data_unique.csv')
    questions = df.Question

    questions = [remove_stopwords_and_punctuation(q) for q in questions]
    lemm_q = get_lemm_questions_series(questions)

    a,b,c=tf_idf(lemm_q,None,None)
    print(f'Broj unique reci u bazi: {len(c)}')


