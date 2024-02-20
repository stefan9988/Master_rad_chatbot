from pathlib import Path

import gensim.downloader
import pandas as pd

from sklearn.metrics import accuracy_score

from util import make_w2v_embeddings, load_quora_questions, prepare_data_for_training, create_malstm_model
from util import split_and_zero_padding

word2vec = gensim.downloader.load('word2vec-google-news-300')

TRAIN_CSV = 'data/train_quora.csv'
TEST_CSV = r'data/test_quora.csv'
MODEL_SAVING_DIR = r'data/best_LSTM_weights.h5'
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 20
N_HIDDEN = 50

# Check if files exist
path_train = Path(TRAIN_CSV)
path_test = Path(TEST_CSV)
if not path_train.is_file() or not path_test.is_file():
    df = load_quora_questions()
    prepare_data_for_training(df)
test_df = pd.read_csv(TEST_CSV)
Y_test = test_df.is_duplicate


test_df, _ = make_w2v_embeddings(test_df, word2vec, embedding_dim=EMBEDDING_DIM)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, MAX_SEQ_LENGTH)
#
# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

model = create_malstm_model(N_HIDDEN, EMBEDDING_DIM, MAX_SEQ_LENGTH)
model.load_weights(MODEL_SAVING_DIR)

preds = model.predict([X_test['left'], X_test['right']])
y_pred = [1 if x > 0.5 else 0 for x in preds]

print(f'Model accuracy: {accuracy_score(Y_test, y_pred)}')
