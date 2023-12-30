from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series
from code.vector_similarity import calculate_similarity

# if __name__ == '__main__':
NUM_FEATURES = 2500

# PREPROCESSING
df = pd.read_csv('data/data_unique.csv')
questions = df.Question
qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
questions_series = get_lemm_questions_series(qs_removed)

asked_question = input('Ask me anything: ')
asked_question = remove_stopwords_and_punctuation(asked_question)
asked_question = get_lemm_questions_series([asked_question])

# TF-IDF
vectorizer = TfidfVectorizer(max_features=NUM_FEATURES)
questions_vectors = vectorizer.fit_transform(questions_series)
asked_question_vector = vectorizer.transform(asked_question)

# SIMILARITY
sim_df = calculate_similarity(questions_vectors, asked_question_vector, questions, 'manhattan')
sim_df = sim_df.iloc[:150, 0]
print(sim_df)
