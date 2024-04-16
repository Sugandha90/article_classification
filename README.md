## Article Classification
In this use case the articles are classified based on the features in the given json file.

## Setup
In the current directory, create the conda environment using:
```
conda create --name test_env python=3.12
```
Activate the environment
```
conda activate test_env
```
Install the requirements from requirements.txt
```
pip install -r requirements.txt
```

## Usage

## Project Folders and their description
### data:
The current folder consists of data folder that contains the following two files:

```
CPS_use_case_classification_training.json: contains the training data
CPS_use_case_classification_response.json: contains the unseen data to generate inference
```

### src:
```
constants.py: contains constants used in the project.
data_preprocessing.py: cleans the prepare the data so that it could be directly fed to ML model. (Takes around 1.5 hrs to complete)
rf_classifier.py: Random forest classification implementation to classify news articels.
script_inference.py: starts the training of defined rf_model.
script_train.py: starts the training of defined rf_model.
utils.py: contains helper functions.
unit_test.py: Contains the unit tests for a few util functions.

```
### rf_logs:
Contains the subfolders for each trained model to store the trained model, training and validation logs, plots, and prediction csv.


### inference:
Contains the response csv and json file with the added category from the trained model ```


### recipes:
Contain the scripts to directly start training and inference using cmd.
Activate the conda env and run:
```
sh recipes/train.sh
sh recipes/inference.sh
```

## Training
1. Start the training of RF model using.
```
python script_train.py
```
2. Training parameters will be saved as gfk/params.json
3. Train, test pickle files will be saved at gfk/data/<filename>.pkl to reuse at later point.

## Evaluation

1. Once Training is done, your model will be saved under /rf_logs/<DATETIME>_train/<DATETIME>_cls
2. Evaluate your model using the following. It uses the parameters from params.json
```
python script_inference.py
```
