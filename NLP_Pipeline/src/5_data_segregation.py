"""
This file aims to segregate data into
train, validation and test files
"""

import logging
import subprocess
import json
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import joblib

def setup_logging():
    """
    Set up the logger to display messages on the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_format = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt='%d-%m-%Y %H:%M:%S')
    c_handler.setFormatter(c_format)
    logger.handlers = [c_handler]
    return logger

def load_config(logger):
    """
    Load configuration from a JSON file.
    """
    try:
        with open('config.json', 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            logger.info('✅ File downloaded with success!')
        return config
    except FileNotFoundError:
        logger.error("❌ Config file not found.")
        return None
    except json.JSONDecodeError:
        logger.error("❌ Error decoding JSON in the config file.")
        return None

def install_libraries(logger):
    """
    Install the required libraries to run fetch_data.py
    """
    try:
        subprocess.run(['pip', 'install', 'wandb'], check=True)
        logger.info('✅ WandB library installed with success!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error installing libraries: {e}")


def download_artifact(logger):
    """
    Download the processed_data artifact from Weights & Biases 
    and load it into a Pandas DataFrame
    """
    try:
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="data_check")
        local_path = run.use_artifact("processed_data:v0").download()
        df = pd.read_csv(f'{local_path}/df_disaster_tweet_processed.csv')
        logger.info('✅ Dataset for data segregation loaded with success!')
        return run, df
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return None, None

def features_and_labels(df, logger):
    """
    Separate features (text) and labels (target)
    in the dataset
    """
    X = df['final']
    y = df['target']
    logger.info(f'Features: {X}')
    logger.info(f'Labels: {y}')
    return X, y

def train_validation_test(X, y, logger):
    """
    Create train, validation and test datasets/arrays
    """
    (train_x, test_x, train_y, test_y) = train_test_split(X, y,test_size=0.2, random_state=42)
    (train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y,test_size=0.2, random_state=42)

    logger.info("Train x: {}".format(train_x.shape))
    logger.info("Train y: {}".format(train_y.shape))
    logger.info("Validation x: {}".format(val_x.shape))
    logger.info("Validation y: {}".format(val_y.shape))
    logger.info("Test x: {}".format(test_x.shape))
    logger.info("Test y: {}".format(test_y.shape))

    return train_x, train_y, val_x, val_y, test_x, test_y

def create_artifacts(train_x, train_y, val_x, val_y, test_x, test_y, run, logger):
    """
    Artifacts to store train, validation and test data
    """

    joblib.dump(train_x, 'train_x')
    joblib.dump(train_y, 'train_y')
    joblib.dump(val_x, 'val_x')
    joblib.dump(val_y, 'val_y')
    joblib.dump(test_x, 'test_x')
    joblib.dump(test_y, 'test_y')

    logger.info("Dumping the train and validation data artifacts to the disk")

    # train_x artifact
    artifact_trainx = wandb.Artifact('train_x', type="train_data", 
                                      description="A json file representing the train_x")
    logger.info("⏳ Logging train_x artifact")
    artifact_trainx.add_file('train_x')
    run.log_artifact(artifact_trainx)
    
    # train_y artifact
    artifact_trainy = wandb.Artifact('train_y', type="train_data", 
                                      description="A json file representing the train_y")
    logger.info("⏳ Logging train_y artifact")
    artifact_trainy.add_file('train_y')
    run.log_artifact(artifact_trainy)

    # val_x artifact
    artifact_valx = wandb.Artifact('val_x', type="val_data", 
                                    description="A json file representing the val_x")
    logger.info("⏳ Logging val_x artifact")
    artifact_valx.add_file('val_x')
    run.log_artifact(artifact_valx)

    # val_y artifact
    artifact_valy = wandb.Artifact('val_y', type="val_data", 
                                    description="A json file representing the val_y")
    logger.info("⏳ Logging val_y artifact")
    artifact_valy.add_file('val_y')
    run.log_artifact(artifact_valy)

    # test_x artifact
    artifact_testx = wandb.Artifact('test_x', type="test_data", 
                                    description="A json file representing the test_x")
    logger.info("⏳ Logging test_x artifact")
    artifact_testx.add_file('test_x')
    run.log_artifact(artifact_testx)

    # test_y artifact
    artifact_testy = wandb.Artifact('test_y', type="test_data", 
                                     description="A json file representing the test_y")
    logger.info("⏳ Logging test_y artifact")
    artifact_testy.add_file('test_y')
    run.log_artifact(artifact_testy)

def data_segregation():
    """
    Main function for data check
    """
    logger = setup_logging()
    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            install_libraries(logger)
            run, df = download_artifact(logger)
            X, y = features_and_labels(df, logger)
            train_x, train_y, val_x, val_y, test_x, test_y = train_validation_test(X, y, logger)
            create_artifacts(train_x, train_y, val_x, val_y, test_x, test_y, run, logger)
            run.finish()
        else:
            logger.error("❌ API key not found in the config file.")
    else:
        logger.error("❌ Unable to load the configuration.")

data_segregation()
