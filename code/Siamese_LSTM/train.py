import pickle
from pathlib import Path

import gensim.downloader
import mlflow
# import mlflow.keras
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from util import make_w2v_embeddings, load_quora_questions, prepare_data_for_training
from util import split_and_zero_padding
from util import create_model

word2vec = gensim.downloader.load('word2vec-google-news-300')

# File paths
TRAIN_CSV = 'data/train_quora.csv'
TEST_CSV = r'data/test_quora.csv'
MODEL_SAVING_DIR = r'data/best_LSTM_weights.h5'
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 20
BATCH_SIZE = 1024
N_EPOCHS = 1
N_HIDDEN = 50

# Check if files exist
path_train = Path(TRAIN_CSV)
path_test = Path(TEST_CSV)
if not path_train.is_file() or not path_test.is_file():
    df = load_quora_questions()
    prepare_data_for_training(df)

# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

train_df, embeddings = make_w2v_embeddings(train_df, word2vec, embedding_dim=EMBEDDING_DIM)
with open('data/embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)

X = train_df[['question1', 'question2']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)

X_train = split_and_zero_padding(X_train, MAX_SEQ_LENGTH)
X_validation = split_and_zero_padding(X_validation, MAX_SEQ_LENGTH)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

model = create_model(N_HIDDEN, EMBEDDING_DIM, MAX_SEQ_LENGTH)

# Set the experiment name
mlflow.set_experiment("LSTM training")
if not mlflow.active_run():
    mlflow.start_run()
tracking_uri = mlflow.get_tracking_uri()
print(f"MLflow is currently running at: {tracking_uri}")

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Log parameters to MLflow
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("n_epoch", N_EPOCHS)
mlflow.log_param("n_hidden", N_HIDDEN)

checkpoint = ModelCheckpoint(MODEL_SAVING_DIR, monitor='val_loss', save_best_only=True, save_weights_only=True,
                             verbose=1)
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                           callbacks=checkpoint)

# Log metrics to MLflow
mlflow.log_metric("final_train_loss", malstm_trained.history['loss'][-1])
mlflow.log_metric("final_val_loss", malstm_trained.history['val_loss'][-1])
mlflow.log_metric("final_train_accuracy", malstm_trained.history['accuracy'][-1])
mlflow.log_metric("final_val_accuracy", malstm_trained.history['val_accuracy'][-1])

# End MLflow run
mlflow.end_run()