import gensim.downloader
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series
from code.vector_similarity import calculate_similarity, load_data
from code.word2vec import vectorize_sentences_sum

# if __name__ == '__main__':
NUM_FEATURES = 2500

# PREPROCESSING
pretrained_model = gensim.downloader.load('word2vec-google-news-300')
questions, _, _ = load_data()
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

# Word2vec and SIMILARITY
q_vec = vectorize_sentences_sum(questions, pretrained_model)
testq_vec = vectorize_sentences_sum(asked_question, pretrained_model)
knn = NearestNeighbors(n_neighbors=150, metric='manhattan').fit(q_vec[1])
_, b = knn.kneighbors(testq_vec[1])

N_questions = questions[b[0]]
print(N_questions)
