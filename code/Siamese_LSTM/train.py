import pickle

import gensim.downloader
import mlflow as mlflow
import mlflow.keras
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
TRAIN_CSV = 'data/train_quora.csv'
MODEL_SAVING_DIR = r'data/best_LSTM_model.h5'
# DF_QUORA_NR = 'data/df_quora_nr.csv'
# EMBEDDINGS = 'data/embeddings.pkl'


# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20

word2vec = gensim.downloader.load('word2vec-google-news-300')
train_df, embeddings = make_w2v_embeddings(train_df,word2vec, embedding_dim=embedding_dim)
with open('data/embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1', 'question2']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# Model variables
batch_size = 1024
n_epoch = 1
n_hidden = 50

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

# Set the experiment name
mlflow.set_experiment("LSTM training")
if not mlflow.active_run():
    mlflow.start_run()
tracking_uri = mlflow.get_tracking_uri()
print(f"MLflow is currently running at: {tracking_uri}")

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Log parameters to MLflow
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("n_epoch", n_epoch)
mlflow.log_param("n_hidden", n_hidden)


# checkpoint = ModelCheckpoint(MODEL_SAVING_DIR, monitor='val_loss', save_best_only=True, verbose=1)
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))#,callbacks=checkpoint
model.save(MODEL_SAVING_DIR)
model.save_weights('data/LSTM_weights.h5')
# Log metrics to MLflow
mlflow.log_metric("final_train_loss", malstm_trained.history['loss'][-1])
mlflow.log_metric("final_val_loss", malstm_trained.history['val_loss'][-1])
mlflow.log_metric("final_train_accuracy", malstm_trained.history['accuracy'][-1])
mlflow.log_metric("final_val_accuracy", malstm_trained.history['val_accuracy'][-1])

# End MLflow run
mlflow.end_run()
# stavio sam da ga savuje umjesto chackpointa