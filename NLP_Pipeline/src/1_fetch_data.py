"""
This file aims to fetch the data required to make a
NLP application with disaster tweets
"""

import subprocess
import json
import logging

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

def fetch_and_organize_data(logger):
    """
    Download dataset files and organize them in 'dataset' directory.
    """
    try:
        # Download of the dataset (train and test)
        train_file = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'
        test_file = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/test.csv'
        subprocess.run(['wget', train_file], check=True)
        subprocess.run(['wget', test_file], check=True)
        subprocess.run(['mkdir', 'dataset'], check=True)
        # Organizing the files in 'dataset' directory
        subprocess.run(['cp', 'train.csv', 'dataset/'], check=True)
        subprocess.run(['cp', 'test.csv', 'dataset/'], check=True)
        logger.info('✅ Fetch data complete!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error fetching and organizing data: {e}")

def store_fetch_data(api_key, logger):
    """
    Login to WandB and store the dataset as a WandB artifact.
    """
    try:
        # Login in WandB
        subprocess.run(['wandb', 'login', '--relogin', api_key], check=True)
        # Store the dataset as an artifact
        subprocess.run(['wandb', 'artifact', 'put',
                        '--name', 'disaster_tweet_classification/dataset',
                        '--type', 'RawData',
                        '--description', 'Natural Language Processing with Disaster Tweets Dataset',
                        'dataset'], check=True)
        logger.info('✅ Fetch data artifact created with success!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error storing and fetching data with WandB: {e}")

def fetch_data():
    logger = setup_logging()
    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            install_libraries(logger)
            fetch_and_organize_data(logger)
            store_fetch_data(file_api_key, logger)
        else:
            logger.error("❌ API key not found in the config file.")
    else:
        logger.error("❌ Unable to load the configuration.")

fetch_data()