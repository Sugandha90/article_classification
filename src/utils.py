# contain all helper functions
import os
import pickle

import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from constants import CURRENT_DIR, COVARIATE_LIST

from transformers import DistilBertTokenizer, DistilBertModel
from urllib.parse import urlparse

import numpy as np
from constants import REDUNDANT_COLUMNS, LABEL_EN_PATH


scaler = MinMaxScaler()


def fix_na_stats(dataframe):
    """
    Fixes the null items and empty strings of dataframe by dropping them.
    :param dataframe: dataframe to be processed
    :return: clean dataframe
    """
    for column_name in dataframe.columns:
        print('\nColumn name:', column_name)

        # Find and print indices of empty values in the column
        empty_indices = dataframe[dataframe[column_name].isna()].index
        print('Indices of null values:', empty_indices)

        # Count empty values in the column
        count_empty_cells = dataframe[column_name].isna().sum()
        print('Count of null values:', count_empty_cells)

        # Drop rows where column value is empty
        dataframe.dropna(subset=[column_name], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)  # reset the index

        if dataframe[column_name].dtype == 'object':
            # Find anf print indices of empty strings in the column
            empty_strings_indices = dataframe[dataframe[column_name].str.strip() == ''].index
            print('Indices of empty strings:', empty_strings_indices)

            # Count empty strings in the column
            count_empty_strings = len(empty_strings_indices)
            print('Count of empty strings:', count_empty_strings)

            # Drop rows where column is empty string
            dataframe = dataframe[dataframe[column_name] != '']
    print('Emtpy values and strings have been removed.')
    return dataframe


def process_date(dataframe, column_name):
    """
    :param dataframe: dataframe to be processed
    :param column_name: name of the column that contains date
    :return: dataframe with new columns extracted from date
    """
    # Convert the date column in dataframe to datetime format
    dataframe['date'] = pd.to_datetime(dataframe[column_name])
    # Extract relevant components such as year, month, day and day of the week from the date
    dataframe['year'] = dataframe[column_name].dt.year
    dataframe[['year_scalar']] = scaler.fit_transform(dataframe[['year']])
    dataframe['month'] = dataframe[column_name].dt.month
    dataframe['day'] = dataframe[column_name].dt.day
    dataframe['day_of_week'] = dataframe[column_name].dt.dayofweek  # here Monday - Sunday: 0 -6
    print('Four new date features have been extracted.')
    return dataframe


def process_urls(dataframe, column_name):
    """
    :param dataframe: dataframe to be processed
    :param column_name: name of the column that contain urls
    :return: dataframe with new columns extracted from urls
    """
    # Extract domain name
    dataframe['domain'] = dataframe[column_name].apply(lambda x: urlparse(x).netloc)
    domain_values = np.unique(dataframe['domain'])

    # Assign values based on conditions
    dataframe['domain_boolean'] = np.where(dataframe['domain'] == domain_values[0], 1,
                                           np.where(dataframe['domain'] == domain_values[1], 2, 0))

    # Calculate URL length
    dataframe['url_length'] = dataframe[column_name].apply(lambda x: len(x))
    dataframe[['url_length_scalar']] = scaler.fit_transform(dataframe[['url_length']])

    # Calculate path length
    dataframe['path_length'] = dataframe[column_name].apply(lambda x: len(urlparse(x).path))
    dataframe[['path_length_scalar']] = scaler.fit_transform(dataframe[['path_length']])
    print('Three few url based features have been extracted.')
    return dataframe


def clean_author(dataframe, column_name):
    """
    Clean the author name column
    :param dataframe: dataframe to be processed
    :param column_name: name of the column that contain authors
    :return: clean dataframe
    """
    dataframe[column_name] = dataframe[column_name].str.split('Contributor').str[0]
    return dataframe


def create_bert_embeddings():
    """
    create the embeddings using BERT
    :return: tokens and embeddings
    """
    # Load pre-trained BERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Tokenize and encode text using BERT tokenizer
    def tokenize_and_encode(text):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return tokens

    # Generate embeddings using BERT model
    def generate_embeddings(tokens):
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embeddings
        return embeddings

    return tokenize_and_encode, generate_embeddings


def process_text(dataframe, columns):
    """
    Add embeddings for text columns as new columns in data frame
    :param dataframe: data frame to be processed
    :param columns: text columns in dataframe
    :return: data frame with text embeddings as new columns
    """
    # Load tokenizer and model
    for column_name in columns:
        tokenizer, model = create_bert_embeddings()
        new_column_tokens = column_name + '_tokens'
        new_column_embeddings = column_name + '_embeddings'
        # Apply tokenization and embedding generation to each row in the DataFrame
        dataframe[new_column_tokens] = dataframe[column_name].apply(tokenizer)
        embeddings = dataframe[new_column_tokens].apply(model)
        bert_embeddings = embeddings.apply(lambda x: np.array(x))
        dataframe[new_column_embeddings] = bert_embeddings.apply(lambda x: x.flatten())
    print('Text embeddings are added for headline and authors.')
    return dataframe


def remove_redundant_columns(dataframe):
    """
    Remove redundant columns from DataFrame based on a list of column names to save storage space.
    :param dataframe: Original dataFrame to remove columns from
    :return: DataFrame with specified columns removed
    """
    dataframe = dataframe.drop(columns=REDUNDANT_COLUMNS, inplace=False)
    print('The columns from which new features have been already extracted are removed')
    return dataframe


def encode_category_to_numeric(dataframe, column_name):
    """
    Encode categories in a DataFrame column to numeric labels.
    :param dataframe: DataFrame containing the column to encode
    :param column_name: Name of the column containing categories
    :return: Encoded DataFrame with numeric labels
    """

    encoder = LabelEncoder()
    encoder.fit(dataframe[column_name])
    y_encoded = encoder.transform(dataframe[column_name])

    with open(LABEL_EN_PATH, 'wb') as pickle_file:
        pickle.dump(encoder, pickle_file)

    dataframe[column_name + '_labels'] = y_encoded
    return dataframe


def select_subset(dataframe):
    """
    Selects the subset of dataset based on the min. count of a category
    :param dataframe: original data frame
    :return: subset data frame
    """
    min_category_count = dataframe['category'].value_counts().min()
    # Group by the category column and apply the sampling function to each group
    dataframe = dataframe.groupby('category', group_keys=False).apply(
        lambda group: sample_from_group(group, min_category_count))
    return dataframe


def remove_similar_category(dataframe):
    """
    Removes the categories that have same meaning
    :param dataframe: dataframe to be processed
    :return: clean dataframe
    """
    dataframe.category = dataframe['category'].map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    dataframe.category = dataframe['category'].map(lambda x: "STYLE & BEAUTY" if x == "STYLE" else x)
    dataframe.category = dataframe['category'].map(lambda x: "ARTS & CULTURE" if x == "ARTS" else x)
    dataframe.category = dataframe['category'].map(lambda x: "ARTS & CULTURE" if x == "CULTURE & ARTS" else x)
    return dataframe


def plot_feature_importance(importances, x_test, path_plot):
    """
    :param importances: array of feature importance from the model
    :param x_test: data frame for test features
    :param path_plot: path to feature importance plot
    :return: Create and save the feature importance plot
    """
    indices = np.argsort(importances)[::-1]
    indices = indices[:12]  # get indices of only top 12 features
    x_axis = importances[indices][::-1]
    idx = indices[::-1]
    y_axis = range(len(x_axis))
    labels = []
    for i in range(len(x_axis)):
        labels.append(x_test.columns[idx[i]])  # get corresponding labels of the features
    y_ticks = np.arange(0, len(x_axis))
    fig, ax = plt.subplots()
    ax.barh(y_axis, x_axis)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels)
    ax.set_title("Random Forest TOP 12 Important Features")
    fig.tight_layout()
    plt.savefig(path_plot, bbox_inches='tight', dpi=400)  # Export in .png file (image)


def plot_confusion_matrix(cm, cm_path):
    """
    :param cm: a confusion matrix of integer classes, (array, shape = [n, n])
    :param cm_path: path to saved confusion matrix
    :return: matplotlib figure containing the plotted confusion matrix.
    """
    # Display the confusion matrix using heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted category')
    plt.ylabel('True category')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path, bbox_inches="tight")


def evaluation(pred_csv_path, validation_log_path):
    """
    :param pred_csv_path: path to saved csv file that contains predictions
    :param validation_log_path:  path to saved log file that contains validation metrics
    :return: Calculates the validation metrics and creates the validation log
    """
    df = pd.read_csv(pred_csv_path)
    val_targ = df['target']
    val_predict = df['predicted_category']

    val_f1_micro = round(f1_score(val_targ, val_predict, average='micro'), 4)
    val_recall_micro = round(recall_score(val_targ, val_predict, average='micro'), 4)
    val_precis_micro = round(precision_score(val_targ, val_predict, average='micro'), 4)
    val_f1_macro = round(f1_score(val_targ, val_predict, average='macro'), 4)
    val_recall_macro = round(recall_score(val_targ, val_predict, average='macro'), 4)
    val_precis_macro = round(precision_score(val_targ, val_predict, average='macro'), 4)
    val_cm = confusion_matrix(val_targ, val_predict)
    accuracy = accuracy_score(val_targ, val_predict)
    balanced_accuracy = balanced_accuracy_score(val_targ, val_predict)
    classwise_accuracy = confusion_matrix(val_targ, val_predict, normalize="true").diagonal()
    print("Please check log at {}  \n".format(validation_log_path))
    with open(validation_log_path, 'a') as f:
        f.writelines(
            "Evaluation metrics on test data \n F1_macro: {} \n F1_micro: {} \n Precision_macro: {} \n "
            "Precision_micro: {} \n Recall_macro: {} \n Recall_micro: {} \n Accuracy: {}"
            "\n Balanced_accuracy: {} \n Class wise accuracy: {}".format(val_f1_macro, val_f1_micro,
                                                                         val_precis_macro, val_precis_micro,
                                                                         val_recall_macro, val_recall_micro,
                                                                         accuracy, balanced_accuracy,
                                                                         classwise_accuracy))
    cm_path = validation_log_path.replace('_validation_log.txt', '_cm.png')

    plot_confusion_matrix(val_cm, cm_path=cm_path)


def sample_from_group(group, n):
    return group.sample(n, random_state=42)


def combine_embeddings_numerics(df_train):
    """
    Combines the embedding and numerical features
    :param df_train: training data frame
    :return: combined features
    """
    x_numerical = df_train[COVARIATE_LIST]
    headline_embeddings_df = pd.DataFrame(np.vstack(df_train['headline_embeddings']))
    headline_embeddings_df.columns = [str(col) + '_headline' for col in headline_embeddings_df.columns]

    authors_embeddings_df = pd.DataFrame(np.vstack(df_train['authors_embeddings']))
    authors_embeddings_df.columns = [str(col) + '_authors' for col in authors_embeddings_df.columns]

    x_conc = pd.concat([x_numerical, headline_embeddings_df, authors_embeddings_df], axis=1)

    return x_conc


def prepare_train_test(x, y):
    """
    Prepare the train and test data
    :param x: preprocessed features
    :param y: labels of preprocessed features
    :return: return pickle file paths of X_train, X_test, y_train, y_test
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Dump to pickle
    x_train_pkl = os.path.join(CURRENT_DIR, 'data/' + 'X_train.pkl')
    x_test_pkl = os.path.join(CURRENT_DIR, 'data/' + 'X_test.pkl')
    y_train_pkl = os.path.join(CURRENT_DIR, 'data/' + 'y_train.pkl')
    y_test_pkl = os.path.join(CURRENT_DIR, 'data/' + 'y_test.pkl')

    x_train.to_pickle(x_train_pkl)
    x_test.to_pickle(x_test_pkl)
    y_train.to_pickle(y_train_pkl)
    y_test.to_pickle(y_test_pkl)

    return x_train_pkl, x_test_pkl, y_train_pkl, y_test_pkl
