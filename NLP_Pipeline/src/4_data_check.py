"""
This file aims to check data related to 
disaster tweets before training
"""

import logging
import subprocess
import json
import pandas as pd
import wandb

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
        logger.info('✅ Dataset for tests loaded with success!')
        return run, df
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return None, None

def test_columns_presence(data, logger):
    """
    Test if the required columns 'text' and 'target' are present in the DataFrame.
    """
    if 'text' in data.columns and 'target' in data.columns:
        logger.info('✅ Test 1 OK!')
        return True
    logger.error('❌ Failed Test 1')
    return False


def test_columns_types(data, logger):
    """
    Test if the data types of 'text' and 'target' columns are as expected.
    """
    if data['text'].dtype == object and data['target'].dtype == int:
        logger.info('✅ Test 2 OK!')
        return True
    logger.error('❌ Failed Test 2')
    return False


def test_data_length(data, logger):
    """
    Test if the length of the DataFrame is greater than 1000.
    """
    if len(data) > 1000:
        logger.info('✅ Test 3 OK!')
        return True
    logger.error('❌ Failed Test 3')
    return False


def data_check():
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
            test1 = test_columns_presence(df, logger)
            test2 = test_columns_types(df, logger)
            test3 = test_data_length(df, logger)
            logger.info(f'{test1+test2+test3}/3 tests passed!')
            run.finish()
        else:
            logger.error("❌ API key not found in the config file.")
    else:
        logger.error("❌ Unable to load the configuration.")

data_check()
