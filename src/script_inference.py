# Clean the response data and use the trained model to generate inferences
import os
import warnings
import json
import pickle
import argparse

import data_preprocessing
from sklearn.exceptions import DataConversionWarning

from constants import RESPONSE_FILE_PATH, PARAM_FILE_PATH, ADDED_FEATURE_COLUMNS, CURRENT_DIR, LABEL_EN_PATH
from utils import combine_embeddings_numerics

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # ArgumentParser object
    # adding the file paths as arguments with default values

    parser.add_argument(
        "--data_path_response_json",
        type=str, help="Enter the path to CPS_use_case_classification_response.json", default=RESPONSE_FILE_PATH)

    # parsing the command line arguments
    args = parser.parse_args()

    # accessing the argument values and get the path to response file
    RESPONSE_FILE_PATH = args.data_path_response_json

    print('Path to CPS_use_case_classification_response.json: ', RESPONSE_FILE_PATH)

    #################################################################################################

    # process the response data to get corresponding features
    response_processor = data_preprocessing.DataPreprocessor(RESPONSE_FILE_PATH, select_subset_flag=False,
                                                             mode='inference')
    df_response = response_processor.process_data()

    # combine the embeddings and other numeric features
    X = combine_embeddings_numerics(df_response)
    
    X_response_pkl = os.path.join(CURRENT_DIR, 'data/' + 'X_response.pkl')
    X.to_pickle(X_response_pkl)

    # load the saved parameters from training
    with open(PARAM_FILE_PATH, 'r') as f:
        params = json.load(f)

    covar_list = params['covar_list']  # list of selected covariates from training
    rf_model_path = params['model_path']  # path to trained random forest model

    # Get the features based on covar_list
    x_test = X[covar_list]

    # load the trained model
    with open(rf_model_path, 'rb') as f:
        classifier = pickle.load(f)

    # Predict on test data set
    prediction = classifier.predict(x_test)

    # save the response to response dataframe
    df_response['predicted_category'] = prediction

    # load the label encodings
    with open(LABEL_EN_PATH, 'rb') as f:
        label_dict = pickle.load(f)

    # convert the encoded labels back to original categories
    df_response['predicted_category_label'] = label_dict.inverse_transform(prediction)

    # drop the additionally added features from data frame before saving to save space
    df_response.drop(columns=ADDED_FEATURE_COLUMNS, axis=1, inplace=True)

    # generating the inference folder to save inferences on response file
    inference_folder = os.path.join(CURRENT_DIR, 'inference')
    if not os.path.exists(inference_folder):
        os.mkdir(inference_folder)

    # name of the model used
    model_name = rf_model_path.split(os.sep)[-1]

    # path to inference
    pred_csv_path = os.path.join(inference_folder, '%s_inference.csv' % model_name)
    df_response.to_csv(pred_csv_path, index=False)
    print('\nPlease check the inference csv in inference folder saved as', pred_csv_path)

    # saving to original json format
    df_response['date'] = df_response['date'].astype(str)  # req. to dump date as a string to json
    df_dict = df_response.to_dict(orient='records')
    # path of new response json
    pred_json_path = os.path.join(inference_folder, 'CPS_use_case_classification_response.json')
    # Dump the dictionary into a JSON file
    with open(pred_json_path, 'w') as json_file:
        json.dump(df_dict, json_file)
    print('\nPlease check the inference json in inference folder saved as', pred_json_path)
