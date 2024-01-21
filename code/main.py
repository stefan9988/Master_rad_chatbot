import gensim.downloader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from code.Siamese_LSTM.util import create_model, make_w2v_embeddings, split_and_zero_padding
from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series
from code.vector_similarity import calculate_similarity, load_data
from code.word2vec import vectorize_sentences_sum
from code.keyword_extraction import extract_keywords

# if __name__ == '__main__':
NUM_FEATURES = 2500
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 20
N_HIDDEN = 50
MODEL_SAVING_DIR = r'data/best_LSTM_weights.h5'
VECTORIZER_FILENAME = r'data/vectorizer.pkl'
XGB_FILENAME = r'data/xgb_model.pkl'

# PREPROCESSING
pretrained_model = gensim.downloader.load('word2vec-google-news-300')
df, _, _ = load_data()
questions = df.Question
answers = df.Answer

qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
questions_series = get_lemm_questions_series(qs_removed)

asked_question = input('Ask me anything: ')
asked_question = remove_stopwords_and_punctuation(asked_question)
asked_question = get_lemm_questions_series([asked_question])

# # TF-IDF
# vectorizer = TfidfVectorizer(max_features=NUM_FEATURES)
# questions_vectors = vectorizer.fit_transform(questions_series)
# asked_question_vector = vectorizer.transform(asked_question)
#
# # SIMILARITY
# sim_df = calculate_similarity(questions_vectors, asked_question_vector, questions, 'manhattan')
# sim_df = sim_df.iloc[:150, 0]
# print(sim_df)
#
# # Word2vec and SIMILARITY
q_vec = vectorize_sentences_sum(questions, pretrained_model)
testq_vec = vectorize_sentences_sum(asked_question, pretrained_model)
knn = NearestNeighbors(n_neighbors=150, metric='manhattan').fit(q_vec[1])
_, b = knn.kneighbors(testq_vec[1])
N_questions = questions_series[b[0]]

# Create a DataFrame
data = {
    'question1': [asked_question[0]]*len(N_questions),
    'question2': N_questions
}
df = pd.DataFrame(data)
df, _ = make_w2v_embeddings(df, pretrained_model, embedding_dim=EMBEDDING_DIM)
df = split_and_zero_padding(df, MAX_SEQ_LENGTH)

# LSTM
model = create_model(N_HIDDEN, EMBEDDING_DIM, MAX_SEQ_LENGTH)
model.load_weights(MODEL_SAVING_DIR)

preds = model.predict([df['left'], df['right']])
most_simmilar_question_index = np.argmax(preds)

top_questions = questions[b[0]]
top_answers = answers[b[0]]

most_similar_question= top_questions.iloc[most_simmilar_question_index]
answer = top_answers.iloc[most_simmilar_question_index]
print(most_similar_question)

# Keyword extraction from answer
keywords = extract_keywords(answer)
for keyword in keywords[:5]:
        answer = answer.replace(keyword, keyword.upper())
print(answer)

# Sentiment classification
answer = remove_stopwords_and_punctuation(answer)
answer = get_lemm_questions_series([answer])

if os.path.exists(VECTORIZER_FILENAME):
    # Load the vectorizer from the file
    with open(VECTORIZER_FILENAME, 'rb') as file:
        loaded_vectorizer = pickle.load(file)

    # Transform the text data using the loaded vectorizer
    answer_tr = loaded_vectorizer.transform(answer)
else:
    answer_tr = f'{VECTORIZER_FILENAME} file not found.'

if os.path.exists(XGB_FILENAME):
    with open(XGB_FILENAME, 'rb') as file:
        xgb_model = pickle.load(file)

    sentiment = xgb_model.predict(answer_tr)
    sentiment = 'Positive' if sentiment[0] > 0.5 else 'Negative'
else:
    sentiment = f'{XGB_FILENAME} file not found.'
print(sentiment)

