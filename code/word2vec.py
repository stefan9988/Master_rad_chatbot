import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords

from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series, get_stemmed_questions_series
from code.vector_similarity import load_data, calculate_similarity
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import gensim.downloader
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def get_word_vector(word: str, model) -> np.ndarray:
    """
    Get the word vector representation of a given word from a word embedding model.

    Parameters:
        word (str): The word for which to retrieve the word vector.
        model (Word2Vec or KeyedVectors): The word embedding model (Word2Vec or KeyedVectors) containing word vectors.
        custom (bool, optional): If True, use the custom model.wv attribute to access word vectors (default).
                                 If False, use the model directly to access word vectors.

    Returns:
        numpy.ndarray: The word vector representation of the input word. If the word is not present in the model,
                       a zero vector with the same dimensionality as the word vectors in the model is returned.
    """
    try:
        wv = model[word.lower()]
    except KeyError:
        wv = np.zeros(model.vector_size)
    return wv


def vectorize_sentence(sentence: str, model) -> list:
    """
    Vectorize a sentence by converting each word in the sentence to its word vector representation.

    Parameters:
        sentence (str): The input sentence to be vectorized.
        model (Word2Vec or KeyedVectors): The word embedding model (Word2Vec or KeyedVectors) containing word vectors.
        custom (bool, optional): If True, use the custom model.wv attribute to access word vectors (default).
                                 If False, use the model directly to access word vectors.

    Returns:
        list: A list of numpy arrays, where each array represents the word vector of a word in the input sentence.
    """

    vec_sent = []
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = tokenizer.tokenize(sentence.lower())
    stop_words = []  # set(stopwords.words('english'))
    for word in word_list:
        if word not in stop_words:
            word_vector = get_word_vector(word, model)
            vec_sent.append(word_vector)
    return vec_sent


def vectorize_sentences_sum(questions: list, model) -> list:
    """
        Vectorize a list of sentences using a word embedding model.

        Args:
            questions (list): A list of strings representing the input questions.
            model: The word embedding model used for vectorization.
            custom (bool, optional): A flag indicating whether custom model is used or not.
                                     Default is True.

        Returns:
            list: A list of two lists, where the first list contains the original questions,
                  and the second list contains the vector representations of the questions.
        """
    q_wv = [[], []]
    for question in questions:
        c = np.array([0 for _ in range(model.vector_size)])
        vs = vectorize_sentence(question, model)

        for word in vs:
            word = np.array(word)
            c = np.add(c, word)
        q_wv[0].append(question)
        q_wv[1].append(c)
    return q_wv


if __name__ == '__main__':
    pretrained_model = gensim.downloader.load('word2vec-google-news-300')
    questions, test_questions, original_question = load_data()
    results = []

    # Preprocessing
    for preprocessing in ['No Preprocessing', 'Lemmatization', 'Stemming']:
        if preprocessing == 'Lemmatization':
            qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
            questions_series = get_lemm_questions_series(qs_removed)

            qs_test_removed = [remove_stopwords_and_punctuation(q) for q in test_questions]
            p_test_questions = get_lemm_questions_series(qs_test_removed)

        elif preprocessing == 'Stemming':
            qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
            questions_series = get_stemmed_questions_series(qs_removed)

            qs_test_removed = [remove_stopwords_and_punctuation(q) for q in test_questions]
            p_test_questions = get_stemmed_questions_series(qs_test_removed)

        else:
            questions_series = questions
            p_test_questions = test_questions

        # Word2vec and Similarity
        for metric in ['cosine', 'euclidean', 'manhattan']:
            q_vec = vectorize_sentences_sum(questions_series, pretrained_model)
            testq_vec = vectorize_sentences_sum(p_test_questions, pretrained_model)
            knn = NearestNeighbors(n_neighbors=150, metric=metric).fit(q_vec[1])
            _, b = knn.kneighbors(testq_vec[1])

            N_questions = []
            c = 0
            for i in range(len(test_questions)):
                N_questions.append(questions[b[i]])
                if (questions[b[i]].eq(original_question[i])).any():
                    c += 1
                else:
                    print(test_questions[i])
                    print(original_question[i])
                    print('====================')
            results.append({'Preprocessing': preprocessing, 'Metric': metric, 'Count': c})

    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Preprocessing', y='Count', hue='Metric', data=df_results)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.title('Word2vec Results')
    plt.xlabel('Preprocessing')
    plt.ylabel('Count')
    plt.show()
