import re

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


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


def tf_idf(train_s: pd.Series, test_s: pd.Series or None, num_features: int or None):
    vectorizer = TfidfVectorizer(max_features=num_features)
    train_vectors = vectorizer.fit_transform(train_s)
    if test_s is not None:
        test_vectors = vectorizer.transform(test_s)
    else:
        test_vectors = None
    feature_names = vectorizer.get_feature_names_out()


    return train_vectors, test_vectors, feature_names

if __name__ == '__main__':
    df = pd.read_csv('data/data_unique.csv')
    questions = df.Question

    questions = [remove_stopwords_and_punctuation(q) for q in questions]
    lemm_q = get_lemm_questions_series(questions)

    a,b,c=tf_idf(lemm_q,None,None)
    print(f'Broj unique reci u bazi: {len(c)}')


