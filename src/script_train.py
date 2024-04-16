# Cleans and prepare the data and starts the training of RF classifier
import argparse
import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning

import data_preprocessing
from constants import TRAIN_FILE_PATH, TARGET_COLUMN
from rf_classifier import rf_classifier
from utils import prepare_train_test, combine_embeddings_numerics

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # ArgumentParser object
    # adding the file paths as arguments with default values
    parser.add_argument(
        "--data_path_training_json",
        type=str, help="Enter the path to CPS_use_case_classification_training.json", default=TRAIN_FILE_PATH)

    # parsing the command line arguments
    args = parser.parse_args()

    # accessing the argument values and set the path train file
    TRAIN_FILE_PATH = args.data_path_training_json

    print('\nPath to CPS_use_case_classification_training.json: ', TRAIN_FILE_PATH)

    #################################################################################################

    # process the train data to get corresponding features
    train_processor = data_preprocessing.DataPreprocessor(TRAIN_FILE_PATH, select_subset_flag=True, mode='train')
    df_train = train_processor.process_data()

    # combine the embeddings and other numeric features
    X = combine_embeddings_numerics(df_train)

    # get the labels
    Y = df_train[TARGET_COLUMN]

    # preparing load the train and test data
    X_train_pkl, X_test_pkl, Y_train_pkl, Y_test_pkl = prepare_train_test(X, Y)

    # reading the pickle
    X_train = pd.read_pickle(X_train_pkl)
    X_test = pd.read_pickle(X_test_pkl)
    y_train = pd.read_pickle(Y_train_pkl)
    y_test = pd.read_pickle(Y_test_pkl)

    # training the random forest classifier
    rf_classifier(X_train, y_train, X_test, y_test)

    print('Training finished')
