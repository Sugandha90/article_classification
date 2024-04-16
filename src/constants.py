# Contains all the constants and path variables
import os

SRC_DIR = os.getcwd()  # path to current directory
CURRENT_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(CURRENT_DIR, 'data')  # path to data folder
TRAIN_FILENAME = 'CPS_use_case_classification_training.json'   # training file name
RESPONSE_FILENAME = 'CPS_use_case_classification_response.json'  # response/testing file name

# Construct the path to the JSON file within the same directory
TRAIN_FILE_PATH = os.path.join(DATA_PATH, TRAIN_FILENAME)  # path to training file
RESPONSE_FILE_PATH = os.path.join(DATA_PATH, RESPONSE_FILENAME)  # path to response/testing file

# Specify the file path where you want to save the JSON file
PARAM_FILE_PATH = os.path.join(CURRENT_DIR, "params.json")

LABEL_EN_PATH = os.path.join(DATA_PATH, "label_encode_dict.pkl")

COVARIATE_LIST = ['year_scalar', 'month', 'day', 'day_of_week', 'domain_boolean', 'url_length_scalar',
                  'path_length_scalar']
TARGET_COLUMN = ['category_labels']

REDUNDANT_COLUMNS = ['headline', 'authors', 'date', 'link', 'domain']
ADDED_FEATURE_COLUMNS = ['year', 'year_scalar', 'month', 'day', 'day_of_week', 'domain', 'domain_boolean',
                         'url_length_scalar', 'url_length', 'path_length_scalar', 'path_length', 'headline_tokens',
                         'headline_embeddings', 'authors_tokens', 'authors_embeddings']

# Define the parameter grid to search
PARAM_GRID = {
    'oob_score': [True],
    'n_estimators': [300, 600, 800, ],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True]
}


# Parameters for baseline experiment
MIN_FIMPORTANCE = 0.001


# Kfold parameter
KFOLD = 3
N_JOBS = -1
