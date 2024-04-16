# Random forest classification implementation, creates rf_model directory in the current directory to save all the logs
import os
import json
import time
import _pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

from utils import plot_feature_importance, evaluation
from constants import PARAM_FILE_PATH, CURRENT_DIR, PARAM_GRID, KFOLD, N_JOBS, MIN_FIMPORTANCE, TARGET_COLUMN


def rf_classifier(X_train, y_train, X_test, y_test):
    """
    :param X_train: training features
    :param y_train: ground truth of X train
    :param X_test: test features
    :param y_test: ground truth of X test
    :return: path to trained RF model
    """

    # Initialize Random Forest classifier
    print("\nStarting training...\n")
    rfmodel = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1, random_state=0)
    sel = SelectFromModel(rfmodel, threshold=MIN_FIMPORTANCE)
    fitted = sel.fit(X_train, y_train)
    feature_idx = fitted.get_support()  # Get list of T/F for covariates for which OOB score is upper the threshold
    list_covar = list(X_train.columns[feature_idx])  # Get list of covariates with the selected features
    x_train = fitted.transform(X_train)  # Update the dataframe with the selected features only

    # Instantiate the grid search model
    print("\nStarting Grid search with cross validation...\n")
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=PARAM_GRID, cv=KFOLD,
                               n_jobs=N_JOBS, verbose=0)
    grid_search.fit(x_train, y_train)  # Fit the grid search to the data
    classifier = grid_search.best_estimator_  # Save the best regressor
    classifier.fit(x_train, y_train)  # Fit the best regressor with the data
    # mean cross-validated score (OOB) and stddev of the best_estimator
    best_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    best_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    rf_model_folder = os.path.join(CURRENT_DIR, "rf_logs")  # path to the folder "rf_model"
    if not os.path.exists(rf_model_folder):
        os.mkdir(rf_model_folder)  # creates rf_logs folder inside the project folder

    model_folder = os.path.join(rf_model_folder, time.strftime("%Y%m%d-%H%M%S_") + 'train')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)  # creates folder inside the rf_logs folder, named as per time stamp and file_name

    model_name = time.strftime("%Y%m%d-%H%M%S_") + 'cls'  # model name
    rf_model_path = os.path.join(model_folder, model_name)  # path to saved model

    # save the best regressor
    with open(rf_model_path, 'wb') as f:
        _pickle.dump(classifier, f)
        f.close()

    # Save the log
    log = ""
    message = 'Parameter grid for Random Forest tuning :\n'
    for key in PARAM_GRID.keys():
        message += '    ' + key + ' : ' + ', '.join([str(i) for i in list(PARAM_GRID[key])]) + '\n'
    message += '    ' + 'min_fimportance' + ' : ' + str(MIN_FIMPORTANCE) + '\n'
    log += message + '\n'

    message = 'Optimized parameters for Random Forest after grid search %s-fold cross-validation tuning :\n' % KFOLD
    for key in grid_search.best_params_.keys():
        message += '    %s : %s' % (key, grid_search.best_params_[key]) + '\n'
    log += message + '\n'

    message = "Mean cross-validated score (OOB) and stddev of the best_estimator : %0.3f (+/-%0.3f)" % (
        best_score, best_std) + '\n'
    log += message + '\n'

    # Print mean OOB and stddev for each set of parameters
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    message = "Mean cross-validated score (OOB) and stddev for every tested set of parameter :\n"
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        message += "%0.3f (+/-%0.03f) for %r" % (mean, std, params) + '\n'
    log += message + '\n'

    # Print final model OOB
    message = 'Final Random Forest model run - internal Out-of-bag score (OOB) : %0.3f' % classifier.oob_score_
    log += message + '\n'

    # Save the log
    fout = open(os.path.join(model_folder, '%s_training_log.txt' % model_name), 'w')
    fout.write(log)
    fout.close()

    # Start the predictions on unseen test data set
    print("\nStarting testing...\n")
    # Get features
    x_test = X_test[list_covar]

    # load the trained model
    with open(rf_model_path, 'rb') as f:
        classifier = _pickle.load(f)

    # Predict on test data set
    prediction = classifier.predict(x_test)

    # Save the prediction
    df_pred = pd.DataFrame()
    df_pred["target"] = y_test[TARGET_COLUMN]
    df_pred['predicted_category'] = prediction

    pred_csv_path = os.path.join(model_folder, '%s_predictions.csv' % model_name)
    df_pred.to_csv(pred_csv_path, index=False)

    # Feature importances
    print("Creation of feature importance plot...\n")
    importances = classifier.feature_importances_  # Save feature importances from the model
    path_plot = os.path.join(model_folder, "%s_RF_feature_importance" % model_name)  # path to saved plot
    plot_feature_importance(importances, x_test, path_plot)
    validation_log_path = os.path.join(model_folder, '%s_validation_log.txt' % model_name)
    evaluation(pred_csv_path, validation_log_path)

    # Create a parameter dict required for inference
    params = {
        "model_path": rf_model_path,
        "covar_list": list_covar
    }

    # Dump the dictionary into a JSON file
    with open(PARAM_FILE_PATH, 'w') as json_file:
        json.dump(params, json_file)
