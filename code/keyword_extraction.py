import nltk
import numpy as np

# nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocess the text by removing punctuation and converting to lowercase
text = "A renters insurance policy will typically provide coverage for your personal property less your policy " \
       "deductible in the event of a covered loss ( fire, smoke, and lightning to name a few). Liability coverage " \
       "is normally also part of the policy that may provide coverage in the event that someone is hurt while in your " \
       "rented premise that you are deemed responsible for. Remember every policy is different. Please read your policy " \
       "completely to understand the coverage provided and any exclusions that there may be or contact your local " \




text = text.lower()

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Use part-of-speech tagging to identify the nouns in the text
tags = nltk.pos_tag(tokens)
nouns = [word for (word, tag) in tags if tag in ["NN", "NNS", "NNP"]]  # Consider both singular and plural nouns

# Use term frequency-inverse document frequency (TF-IDF) analysis to rank the nouns
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([text])

# Convert the TF-IDF matrix to a dense NumPy array
dense_tfidf = tfidf.todense()

# Get the indices of the top 3 most important nouns
top_word_indices = np.argsort(dense_tfidf)[0, ::-1]

# Map the indices to the actual nouns
feature_names = vectorizer.get_feature_names_out()
top_words = [feature_names[index] for index in top_word_indices]

top_nouns = [word for word in top_words[0][0] if word in nouns]
print(top_nouns)


