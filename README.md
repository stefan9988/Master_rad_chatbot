# Folder code

## create_unique_df.py 
 - The insurance qna dataset contains some repeating questions with different answers. 
 - This script creates a DataFrame with unique question values and combines all the answers related to the same question.

## preprocessing.py 
 - Contains a function for removing stop words and characters, as well as functions for stemming, 
   lemmatization, and functions related to TF-IDF.

## vector_similarity.py 
 - This script has a function that loads the data and a function that calculates similarities between vectors. 
 - When run it will use TF-IDF word vectors and will show the results for every combination of hyperparameters, 
   such as preprocessing, metrics and number of features.

## word2vec.py 
 - This script is similar to vector_similarity.py, but it uses pretrained model for getting word vectors.
 - It has functions for transforming sentences into vectors. As well as vector_similarity.py this script
   will also show the results.

## keyword_extraction.py 
 - Keywords could be extracted using term frequency, so this script implements code that finds the most frequent 
   nouns in a document and prints the top 5 of them.

## main.py 
 - This script contains the complete code for running the model with the best parameters.

## Siamese_LSTM folder
 - This folder is where all scripts for creating, training and predicting of an LSTM model are. 

