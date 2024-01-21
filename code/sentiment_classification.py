from pathlib import Path
import os
import pickle

import scipy
import shap
import mlflow
import mlflow.keras
import mlflow.xgboost
import mlflow.sklearn
import seaborn as sns
from keras import Sequential
from keras.layers import Dense
from mlflow.models import infer_signature
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

from code.preprocessing import remove_stopwords_and_punctuation, get_lemm_questions_series, tf_idf, \
    assign_sentiment_values


def preprocess_IMDB(IMDB_DATASET):
    df = pd.read_csv(IMDB_DATASET)
    reviews = df.loc[:, 'review']
    df = assign_sentiment_values(df)

    reviews = [remove_stopwords_and_punctuation(r) for r in reviews]
    lemm_r = get_lemm_questions_series(reviews)
    df.review = lemm_r
    df.to_csv(IMDB_LEM, index=False)


def split_data(df: pd.DataFrame, num_features: int):
    """
    Split and preprocess the input DataFrame for machine learning.

    This function takes a DataFrame containing text reviews and sentiment labels,
    splits it into training and testing sets, and applies TF-IDF vectorization to the text data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing text reviews and sentiment labels.

    Returns:
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, List[str]]:
        - X_train_: TF-IDF vectors of the training text reviews.
        - X_test_: TF-IDF vectors of the testing text reviews.
        - y_train: Sentiment labels for the training data.
        - y_test: Sentiment labels for the testing data.
        - feature_names: A list of feature names corresponding to the TF-IDF vectors.

    Note:
    The function assumes that the DataFrame contains 'review' and 'sentiment' columns.
    It also relies on the 'data_preprocessing.tf_idf_1' function for TF-IDF vectorization.
    Ensure you have imported the necessary libraries and modules for text preprocessing
    and machine learning before using this function. The 'num_features' parameter
    should be defined and set appropriately for the TF-IDF vectorization step.
    """

    X = df.review
    y = df.sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_, X_test_, feature_names = tf_idf(X_train, X_test, num_features)

    return X_train_, X_test_, y_train, y_test, feature_names


def csrMatrix2pdDataFrame(csr_matrix: scipy.sparse.csr_matrix) -> pd.DataFrame:
    """
    Convert a CSR matrix to a Pandas DataFrame.

    This function takes a sparse CSR matrix and converts it into a Pandas DataFrame.

    Parameters:
    csr_matrix (scipy.sparse.csr_matrix): The CSR matrix to be converted.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the data from the CSR matrix.

    Note:
    The function is designed to work with CSR matrices, which are commonly used
    for sparse data representations. Ensure that the input matrix is of the correct
    format before using this function.
    """

    df = pd.DataFrame.sparse.from_spmatrix(csr_matrix)
    return df


def print_feature_importances(importance, feature_names, N, name):
    """
        Print the top N most important features and their importance scores.

        This function takes a list of feature importances, their corresponding feature names,
        and a value N, and prints the top N most important features and their importance scores.

        Parameters:
        importance (List[float]): A list of feature importance scores.
        feature_names (List[str]): A list of feature names corresponding to the importance scores.
        N (int): The number of top features to print.
        name (str): The name or identifier for the set of features being printed.

        Note:
        - The input lists 'importance' and 'feature_names' should have the same length.
        - If N is greater than the number of available features, the function will print all available features.
        """
    if N > len(importance):
        N = len(importance)
    feature_importance_dict = dict(zip(feature_names, importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {N} Most Important Features for {name}:")
    for feature, importance in sorted_features[:N]:
        print(f"{feature}: {importance}")


def shap_values(model, X_train, feature_names):
    """
    Calculate and visualize SHAP values.

    This function calculates SHAP (SHapley Additive exPlanations) values for a model
    and visualizes them using a beeswarm plot.

    Parameters:
    model: The trained model for which SHAP values are to be calculated.
    X_train: The training dataset used to train the model.
    feature_names: A list of feature names corresponding to the columns in the training dataset.

    Note:
    - The function relies on the 'shap' library for SHAP value calculation and visualization.
    - Ensure that 'shap' and 'xgboost' libraries are correctly installed and imported.
    - The 'feature_names' parameter is required for the proper interpretation of SHAP values.
    """

    df = X_train

    explainer_xgb = shap.Explainer(model)
    shap_values_xgb = explainer_xgb(df)

    # Visualize SHAP values using a beeswarm plot
    shap.plots.beeswarm(shap_values_xgb)


def evaluate_model(y_test, y_pred, classifier, num_features=0):
    """
    Evaluate a machine learning model's performance and log metrics with MLflow.

    This function assesses the performance of a machine learning model by calculating
    accuracy, confusion matrix, and classification report. It also logs these metrics
    using MLflow for tracking and monitoring.

    Parameters:
    y_test: True labels from the testing dataset.
    y_pred: Predicted labels generated by the model.
    classifier: A string identifying the classifier used (e.g., 'XGBoost', 'Random Forest').
    preprocessing: A string identifying the data preprocessing method (e.g., 'stem', 'lemm').
    num_features: An integer representing the number of features used in the preprocessing.

    Returns:
    float: The accuracy of the model's predictions.

    Note:
    - The function logs various metrics with MLflow, enabling easy tracking and comparison of model performance.
    - Ensure that you have imported and configured MLflow for this function to work correctly.
    - The function assumes that you have imported relevant libraries for model evaluation.
    """

    print(classifier)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    name = classifier + '_' + str(num_features)
    with mlflow.start_run(run_name=name):
        # Log the metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("TP", conf_matrix[0][0])
        mlflow.log_metric("FP", conf_matrix[0][1])
        mlflow.log_metric("FN", conf_matrix[1][0])
        mlflow.log_metric("TN", conf_matrix[1][1])
        mlflow.log_metric("0 precision", classification_rep['0']['precision'])
        mlflow.log_metric("0 precision", classification_rep['0']['recall'])
        mlflow.log_metric("0 precision", classification_rep['0']['f1-score'])
        mlflow.log_metric("0 precision", classification_rep['0']['support'])
        mlflow.log_metric("1 precision", classification_rep['1']['precision'])
        mlflow.log_metric("1 precision", classification_rep['1']['recall'])
        mlflow.log_metric("1 precision", classification_rep['1']['f1-score'])
        mlflow.log_metric("1 precision", classification_rep['1']['support'])
        mlflow.log_metric("macro avg f1", classification_rep['macro avg']['f1-score'])
        mlflow.log_metric("weighted avg f1", classification_rep['weighted avg']['f1-score'])

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)
    print('==========================================')

    return accuracy


def svm_classifier(X_train, X_test, y_train, y_test, num_features):
    """
        Train and evaluate a Support Vector Machine (SVM) classifier.

        Parameters:
        X_train: Training data features.
        X_test: Testing data features.
        y_train: Training data labels.
        y_test: Testing data labels.
        feature_names: List of feature names.
        preprocessing: A string identifying the data preprocessing method (e.g., 'TF-IDF', 'Count Vectorization').
        num_features: An integer representing the number of features used in the preprocessing.

        Returns:
        float: The accuracy of the SVM classifier.

        Note:
        - The function uses a linear kernel for simplicity.
        - The 'evaluate_model' function is used to assess model performance and log metrics.
        - Ensure that the necessary libraries (e.g., scikit-learn) are imported before using this function.
        - Make sure 'preprocessing' and 'num_features' are set appropriately for the evaluation.
        """
    svm_model = SVC(kernel='linear', C=1.0)  # Linear kernel for simplicity
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    acc = evaluate_model(y_test, y_pred, 'SVM', num_features=num_features)

    return acc


def knn_classifier(X_train, X_test, y_train, y_test, num_features):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = knn_model.predict(X_test)

    acc = evaluate_model(y_test, y_pred, 'KNN', num_features=num_features)

    return acc


def xgboost_classifier(X_train, X_test, y_train, y_test, feature_names, num_features):
    xgb_model = XGBClassifier(colsample_bytree=0.8, learning_rate=0.2, n_estimators=750,
                              max_depth=7, min_child_weight=5, subsample=0.7)
    xgb_model.fit(X_train, y_train)

    shap_values(xgb_model, X_train, feature_names)
    importance = xgb_model.feature_importances_
    print_feature_importances(importance, feature_names, 10, 'XGB')

    y_pred = xgb_model.predict(X_test)

    acc = evaluate_model(y_test, y_pred, 'XGBoost', num_features=num_features)

    filename = r'data/xgb_model.pkl'
    if not os.path.exists(filename):
        with open(filename, 'wb') as file:
            pickle.dump(xgb_model, file)

    return acc


def random_forest_classifier(X_train, X_test, y_train, y_test, feature_names, num_features):
    rf_model = RandomForestClassifier(max_depth=33, min_samples_leaf=8, min_samples_split=16,
                                      n_estimators=161, random_state=42)
    rf_model.fit(X_train, y_train)
    # importance = rf_model.feature_importances_
    # print_feature_importances(importance, feature_names, 10, 'Random Forest')

    y_pred = rf_model.predict(X_test)

    acc = evaluate_model(y_test, y_pred, 'Random Forest', num_features=num_features)

    return acc


def neural_network(X_train, X_test, y_train, y_test, num_features):
    X_train = csrMatrix2pdDataFrame(X_train)
    X_test = csrMatrix2pdDataFrame(X_test)

    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    epochs = 100
    batch_size = 32

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    acc = evaluate_model(y_test, y_pred, 'Neural Network', num_features=num_features)

    return acc


def xgb_hyperparameter_tuning(X: pd.Series, y: pd.Series):
    """
    Perform hyperparameter tuning for an XGBoost classifier and evaluate the best model.

    This function splits the data into training, validation, and testing sets, conducts a grid search
    over a range of hyperparameters to find the best combination, trains the XGBoost classifier with
    the best hyperparameters, and evaluates its performance.

    Parameters:
    X (pd.Series): The feature data.
    y (pd.Series): The corresponding target labels.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7],  # [7], Maximum depth of the individual trees in the ensemble.
        'min_child_weight': [1, 3, 5],  # [5],  Minimum hessian needed in a child.
        'subsample': [0.7, 0.8, 0.9],  # [0.7],  Determines the fraction of samples used for growing trees.
        'colsample_bytree': [0.7, 0.8, 0.9],  # [0.8], Add randomness to the feature selection process.
        'learning_rate': [0.01, 0.1, 0.2],  # [0.2], Determines the step size while moving toward a minimum.
        'n_estimators': list(range(100, 1000, 100))  # [700], Number of trees

    }
    xgb_classifier = XGBClassifier()
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_xgb_classifier = XGBClassifier(**best_params)
    best_xgb_classifier.fit(X_train, y_train)
    y_pred = best_xgb_classifier.predict(X_val)
    evaluate_model(y_val, y_pred, 'XGB')


def rf_hyperparameter_tuning(X: pd.Series, y: pd.Series):
    """
    Perform hyperparameter tuning for a Random Forest classifier and evaluate the best model.

    This function splits the data into training, validation, and testing sets, conducts a randomized search
    over a search space of hyperparameters to find the best combination, trains the Random Forest classifier
    with the best hyperparameters, and evaluates its performance.

    Parameters:
    X (pd.Series): The feature data.
    y (pd.Series): The corresponding target labels.

    Note:
    - The data is split into training, validation, and testing sets.
    - Hyperparameters and their search space are defined in 'param_dist'.
    - RandomizedSearchCV is used to find the best hyperparameters based on cross-validated accuracy.
    - The best model is trained and evaluated using the 'evaluate_model' function.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    model = RandomForestClassifier()

    # Define hyperparameters and their search space
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20)
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=1,
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    # Perform the random search
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best hyperparameters found:")
    print(random_search.best_params_)

    # Best Hyperparameters:
    # {'max_depth': 33, 'min_samples_leaf': 8, 'min_samples_split': 16, 'n_estimators': 161}

    best_rf_classifier = RandomForestClassifier(**random_search.best_params_)
    best_rf_classifier.fit(X_train, y_train)
    y_pred = best_rf_classifier.predict(X_val)
    evaluate_model(y_val, y_pred, 'Random Forest')


def plot_results(df_results):
    """
    Plot and compare accuracy results of different algorithms with lemmatization.

    This function creates a grouped bar chart to visually compare the accuracy of multiple
    algorithms with lemmatization.

    Parameters:
    df_results (pd.Dataframe): Dataframe that contains results data
    Note:
    - The function creates a grouped bar chart to visualize accuracy results for different algorithms.
    - The accuracy values are displayed above the bars in the chart.
    """

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='NumFeatures', y='Acc', hue='Alg', data=df_results)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.title(f'Sentiment Classification Results')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')

    save_path = f"results/Sentiment_Classification.png"
    plt.savefig(save_path)


if __name__ == '__main__':
    IMDB_DATASET = r'data/IMDB Dataset.csv'
    IMDB_LEM = 'data/IMDB_lemm.csv'

    path_imdb_lem = Path(IMDB_LEM)
    if not path_imdb_lem.is_file():
        preprocess_IMDB(IMDB_DATASET)

    df = pd.read_csv(IMDB_LEM)
    results = []

    for num_features in [100, 500, 1000, 2000]:
        X_train, X_test, y_train, y_test, feature_names = split_data(df, num_features)

        a1 = svm_classifier(X_train, X_test, y_train, y_test, num_features)
        a2 = knn_classifier(X_train, X_test, y_train, y_test, num_features)
        a3 = xgboost_classifier(X_train, X_test, y_train, y_test, feature_names, num_features)
        a4 = random_forest_classifier(X_train, X_test, y_train, y_test, feature_names, num_features)
        a5 = neural_network(X_train, X_test, y_train, y_test, num_features)

        acc_lemm = [a1, a2, a3, a4, a5]
        alg_names = ['SVM', 'KNN', 'XGB', 'RF', 'NN']
        for i in range(5):
            results.append({'Alg': alg_names[i], 'NumFeatures': num_features, 'Acc': round(acc_lemm[i], 2)})

    df_results = pd.DataFrame(results)

    plot_results(df_results)

    # xgb_hyperparamether_tuning(X, y)
    # rf_hyperparamether_tuning(X,y)
