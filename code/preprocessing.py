import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer


def remove_stopwords_and_punctuation(question):
    word_list = nltk.word_tokenize(question.lower())
    stop = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    pattern = r'[^\w\s]'

    processed_words = [re.sub(pattern, '', word) for word in word_list if word not in stop]
    processed_question = ' '.join(processed_words)

    return processed_question


def lemm(question):
    word_list = nltk.word_tokenize(question)
    lemmatizer = WordNetLemmatizer()

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


def stem(question):
    word_list = nltk.word_tokenize(question)
    stemmer = PorterStemmer()

    stemmed_output = ' '.join([stemmer.stem(w) for w in word_list])
    return stemmed_output


def get_stemmed_questions_series(questions):
    stemmed_questions = pd.Series([stem(q) for q in questions], name='Question')
    return stemmed_questions


def get_lemm_questions_series(questions):
    lemm_questions = pd.Series([lemm(q) for q in questions], name='Question')
    return lemm_questions


if __name__ == '__main__':
    df = pd.read_csv('data/data_unique.csv')
    questions = df.Question

    questions = [remove_stopwords_and_punctuation(q) for q in questions]
    print(get_lemm_questions_series(questions))
