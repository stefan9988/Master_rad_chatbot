import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series, tf_idf, \
    get_stemmed_questions_series
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """
        Load data from the 'data/test_questions_json.json' file and a CSV file ('data/data_unique.csv').

        Returns:
        - pd.Series: Series of questions from the CSV file.
        - pd.Series: Series of test questions from the JSON file.
        - pd.Series: Series of original questions from the JSON file.
        """
    with open(r'data/test_questions_json.json', 'r') as file:
        data = json.load(file)

    test_questions = data['question']
    original_question = data['original']

    df = pd.read_csv('data/data_unique.csv')
    questions = df.Question

    return questions, pd.Series(test_questions, name='Test Question'), pd.Series(original_question,
                                                                                 name='Original Question')


def calculate_similarity(all_questions_vector, question_vector, questions, metric='cosine') -> pd.DataFrame:
    """
        Calculate similarity or distance between a question vector and all question vectors.

        Parameters:
        - all_questions_vector: Matrix of vectors representing all questions.
        - question_vector: Vector representing the target question.
        - questions: Series of questions corresponding to the vectors.
        - metric (str): Similarity or distance metric to be used ('cosine' or any valid metric supported by sklearn).

        Returns:
        - pd.DataFrame: DataFrame containing questions and their similarity or distance values.
        """
    if metric == 'cosine':
        similarities = cosine_similarity(question_vector, all_questions_vector).flatten()
        df = pd.DataFrame({'Question': questions, 'Cosine': similarities.flatten()})
        df.sort_values('Cosine', ascending=False, inplace=True)
    else:
        distance = pairwise_distances(all_questions_vector, question_vector, metric=metric)
        df = pd.DataFrame({'Question': questions, metric: distance.flatten()})
        df.sort_values(metric, inplace=True)

    return df


if __name__ == '__main__':
    # Load df
    questions, test_questions, original_question = load_data()

    # Preprocessing
    for preprocessing in ['No Preprocessing', 'Lemmatization', 'Stemming']:
        if preprocessing == 'Lemmatization':
            qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
            questions_series = get_lemm_questions_series(qs_removed)

            qs_test_removed = [remove_stopwords_and_punctuation(q) for q in test_questions]
            p_test_questions = get_lemm_questions_series(qs_test_removed)

        elif preprocessing == 'Stemming':
            qs_removed = [remove_stopwords_and_punctuation(q) for q in questions]
            questions_series = get_stemmed_questions_series(qs_removed)

            qs_test_removed = [remove_stopwords_and_punctuation(q) for q in test_questions]
            p_test_questions = get_stemmed_questions_series(qs_test_removed)

        else:
            questions_series = questions
            p_test_questions = test_questions

        # TF-IDF and Similarity
        results = []
        for metric in ['cosine', 'euclidean', 'manhattan']:
            for num_features in [500, 1000, 2500, 3162]:  # 3162 jedinstvene reci u bazi

                train, test, features = tf_idf(questions_series, p_test_questions, num_features)

                c = 0
                for i in range(len(original_question)):
                    sim_df = calculate_similarity(train, test[i], questions, metric)
                    if (sim_df.iloc[:150, 0].eq(original_question[i])).any():
                        c += 1
                results.append({'Metric': metric, 'NumFeatures': num_features, 'Count': c})

        df_results = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='NumFeatures', y='Count', hue='Metric', data=df_results)
        for i in ax.containers:
            ax.bar_label(i, )
        plt.title(f'{preprocessing} Results')
        plt.xlabel('Number of Features')
        plt.ylabel('Count')
        plt.show()
