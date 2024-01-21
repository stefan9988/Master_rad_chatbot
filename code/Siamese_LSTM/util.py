import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
import gensim.downloader
import numpy as np
import itertools

from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series


def load_quora_questions() -> pd.DataFrame:
    """
    Load Quora questions from a TSV (Tab-Separated Values) file into a pandas DataFrame and removes
    all rows with null values.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the Quora questions data.
    """
    tsv_file_path = r'data/quora_duplicate_questions.tsv'

    with open(tsv_file_path, mode='r', encoding='utf-8') as tsv_file:
        tsv_data = [line.strip().split('\t') for line in tsv_file]

    df = pd.DataFrame(tsv_data[1:], columns=tsv_data[0])
    df.dropna(inplace=True)
    return df.iloc[:, 3:]


def prepare_data_for_training(df):
    """
    Preprocesses and prepares data for training a machine learning model.

    This function loads Quora questions from a DataFrame, removes stopwords, lemmatize questions
    and splits the data into training and testing sets. It saves the preprocessed data
    into CSV files for further use.

    Returns:
        None
    """
    question1 = df.question1
    question2 = df.question2

    question1 = [remove_stopwords_and_punctuation(q) for q in question1]
    question2 = [remove_stopwords_and_punctuation(q) for q in question2]

    lemm_q1 = get_lemm_questions_series(question1)
    lemm_q2 = get_lemm_questions_series(question2)

    df['question1'] = lemm_q1
    df['question2'] = lemm_q2

    X = df.drop(columns=['is_duplicate'])
    y = df['is_duplicate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv('data/train_quora.csv', index=False)
    test_df.to_csv('data/test_quora.csv', index=False)



def text_to_word_list(text):
    """
    Preprocess and convert a text string to a list of words.

    Parameters:
    - text (str): Input text to be processed.

    Returns:
    - list: A list of words extracted from the input text.
    """
    text = str(text)
    text = text.lower()
    text = text.split()

    return text


def make_w2v_embeddings(df, word2vec, embedding_dim=300):
    """
    Embeds text data in a DataFrame using pre-trained word embeddings.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'question1' and 'question2' columns.
    - word2vec: Pre-trained word2vec model.
    - embedding_dim (int): Dimensionality of word embeddings.

    Returns:
    - tuple: DataFrame with embedded representations and the embedding matrix.
    """
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Stopwords
    stops = set(stopwords.words('english'))

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 50000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for question in ['question1', 'question2']:

            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):
                # Check for unwanted words
                if word in stops:
                    continue

                # If a word is missing from word2vec model.
                if word not in word2vec.key_to_index:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append question as number representation
            df.at[index, question] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.key_to_index:
            embeddings[index] = word2vec[word]
    del word2vec

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):
    """
    Split text data in a DataFrame and perform zero-padding for sequence alignment.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'question1' and 'question2' columns.
    - max_seq_length (int): Maximum sequence length for padding.

    Returns:
    - dict: A dictionary with 'left' and 'right' keys, each containing padded sequences.
    """
    X = {'left': df['question1'], 'right': df['question2']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


def create_model(n_hidden, embedding_dim, max_seq_length):
    """
    Create a Siamese LSTM model for text similarity prediction.

    Parameters:
    - n_hidden (int): Number of LSTM units in the hidden layer.
    - embedding_dim (int): Dimensionality of word embeddings.
    - max_seq_length (int): Maximum sequence length for input data.

    Returns:
    - keras.models.Model: A Siamese LSTM model for text similarity prediction.
    """
    with open('data/embeddings.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

    # LSTM
    x.add(LSTM(n_hidden))

    shared_model = x

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    return model


class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# class EmptyWord2Vec:
#     """
#     Just for test use.
#     """
#     vocab = {}
#     word_vec = {}


