from pathlib import Path
import gensim.downloader
from keras.utils import pad_sequences

# from keras.src.utils import pad_sequences
from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series
from time import time
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec, KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import json
import itertools
import datetime
import gensim.downloader
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers import Layer


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




def get_train_test_validation_data(train_df, test_df):
    def text_to_word_list(text):
        text = str(text)
        text = text.lower()
        text = text.split()

        return text

    # Prepare embedding
    vocabulary = dict()
    inverse_vocabulary = [
        '<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = gensim.downloader.load('word2vec-google-news-300')
    #TODO izbaci ga napolje
    questions_cols = ['question1', 'question2']

    # Iterate over the questions only of both training and test datasets
    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for question in questions_cols:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):

                    # Check for unwanted words
                    if word not in word2vec:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representation
                dataset.at[index, question] = q2n

    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec:
            embeddings[index] = word2vec[word]

    filepath = 'data/vocabulary.pkl'
    path = Path(filepath)
    if not path.is_file():
        with open(filepath, 'wb') as file:
            pickle.dump(vocabulary, file)

    max_seq_length = 20
    # Split to train validation
    validation_size = 40000
    training_size = len(train_df) - validation_size

    questions_cols = ['question1', 'question2']

    X = train_df[questions_cols]
    Y = train_df['is_duplicate']
    Y_test = test_df['is_duplicate']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.question1, 'right': X_train.question2}
    X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
    X_test = {'left': test_df.question1, 'right': test_df.question2}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values
    Y_test = Y_test.values

    # # Zero padding
    for dataset, side in itertools.product([X_train, X_test, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test, max_seq_length, embeddings, embedding_dim


def train_LSTM(X_train, Y_train, X_validation, Y_validation, max_seq_length, embeddings, embedding_dim,
               MODEL_SAVING_DIR):
    # Model variables
    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 500
    n_epoch = 10

    def exponent_neg_manhattan_distance(left, right):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                                trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden, return_sequences=True)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                             output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])

    # Save the model architecture as JSON
    with open(r'data/LSTM_architecture.json',
              'w') as f:
        f.write(malstm.to_json())

    # Model checkpoint to save the best model during training
    checkpoint = ModelCheckpoint(MODEL_SAVING_DIR, monitor='val_loss', save_best_only=True, verbose=1)

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(lr=0.01, clipnorm=gradient_clipping_norm)

    malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    # Start training
    training_start_time = time()

    malstm_trained = malstm.fit([X_train['left'], X_train['right']],
                                Y_train, batch_size=batch_size,
                                epochs=n_epoch,
                                validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                                callbacks=[checkpoint])

    print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                            datetime.timedelta(
                                                                seconds=time() - training_start_time)))
    return malstm


if __name__ == '__main__':
    TRAIN_CSV = r'data/train_quora.csv'
    TEST_CSV = r'data/test_quora.csv'
    MODEL_SAVING_DIR = r'data/best_LSTM_model.h5'

    path_train = Path(TRAIN_CSV)
    path_test = Path(TEST_CSV)
    if not path_train.is_file() or not path_test.is_file():
        df = load_quora_questions()
        prepare_data_for_training(df)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, Y_train, X_validation, Y_validation, X_test, Y_test, max_seq_length, embeddings, embedding_dim = get_train_test_validation_data(
        train_df, test_df)

    malstm = train_LSTM(X_train, Y_train, X_validation, Y_validation, max_seq_length, embeddings, embedding_dim,
                        MODEL_SAVING_DIR)
    preds = malstm.predict([X_test['left'], X_test['right']])
    y_pred = [1 if x > 0.5 else 0 for x in preds]

    print(f'Model accuracy: {accuracy_score(Y_test, y_pred)}')
