import pickle

import gensim.downloader
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from keras.models import load_model
from sklearn.metrics import accuracy_score

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# # File paths
# TEST_CSV = 'data/test-20.csv'
#
# # Load training set
# test_df = pd.read_csv(TEST_CSV)
# for q in ['question1', 'question2']:
#     test_df[q] = test_df[q]
#
# # Make word2vec embeddings
# embedding_dim = 300
# max_seq_length = 20
# test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim)
#
# # Split to dicts and append zero padding.
# X_test = split_and_zero_padding(test_df, max_seq_length)
#
# # Make sure everything is ok
# assert X_test['left'].shape == X_test['right'].shape
#
# # --
#
# model = tf.keras.models.load_model('data/best_LSTM_model.h5', custom_objects={'ManDist': ManDist})
# model.summary()
#
# prediction = model.predict([X_test['left'], X_test['right']])
# print(prediction)

TEST_CSV = r'data/test_quora.csv'
test_df = pd.read_csv(TEST_CSV)
Y_test = test_df.is_duplicate
word2vec = gensim.downloader.load('word2vec-google-news-300')

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, _ = make_w2v_embeddings(test_df,word2vec, embedding_dim=embedding_dim)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)
#
# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape
# model = load_model('data/best_LSTM_model.h5', custom_objects={'ManDist': ManDist}, compile=False)
# model = load_model('data/best_LSTM_model.h5', custom_objects={'ManDist': ManDist})
# print(model.summary())

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM


n_hidden = 50
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
model.load_weights('data/LSTM_weights.h5')

preds = model.predict([X_test['left'], X_test['right']])
y_pred = [1 if x > 0.5 else 0 for x in preds]

print(f'Model accuracy: {accuracy_score(Y_test, y_pred)}')