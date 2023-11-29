"""
This file aims to preprocess textual data related to 
disaster tweets and generate illustrative word clouds.
"""

import re
import json
import logging
import subprocess
import nltk
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud

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
    Install the required libraries to run preprocessing.py
    """
    try:
        subprocess.run(['pip', 'install', 'wandb'], check=True)
        logger.info('✅ WandB library installed with success!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error installing libraries: {e}")

def download_dataset_artifact(logger):
    """
    Initialize a wandb run and download "dataset" artifact
    """
    try:
        # Initialize wandb run
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="preprocessing")

        # Get the artifact
        artifact = run.use_artifact('dataset:v0')

        # Download the content of the artifact to the local directory
        artifact_dir = artifact.download()

        # Path to dataset
        data_path = artifact_dir + '/train.csv'

        # Load dataset
        df_disaster_tweet = pd.read_csv(data_path)
        logger.info('✅ Artifact downloaded with success!')

        return df_disaster_tweet, run
    except Exception as e:
        logger.error(f"❌ Error during artifact download: {e}")
        return None, None

def punctuations(text):
    """
    Removes non-alphabetic characters from the input text.
    """
    return re.sub(r'[^a-zA-Z]', ' ', text)

def tokenization(text):
    """
    Tokenizes the input text into individual words.
    """
    return word_tokenize(text)

def stopwords_remove(tokens):
    """
    Removes common English stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')  # Preserve 'not' for sentiment analysis
    return [word for word in tokens if word not in stop_words]

def lemmatization(tokens):
    """
    Performs lemmatization on a list of tokens using WordNet lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word=word, pos='v') for word in tokens]

def text_preprocessing(df_disaster_tweet, logger):
    """
    Performs text preprocessing on the 'text' column of the given DataFrame.
    """
    try:
        df_disaster_tweet['text_lower'] = df_disaster_tweet['text'].str.lower()
        df_disaster_tweet['text_no_punct'] = df_disaster_tweet['text_lower'].apply(punctuations)
        df_disaster_tweet['text_tokenized'] = df_disaster_tweet['text_no_punct'].apply(tokenization)
        df_disaster_tweet['text_no_stop'] = df_disaster_tweet['text_tokenized'].apply(stopwords_remove)
        df_disaster_tweet['text_lemmatized'] = df_disaster_tweet['text_no_stop'].apply(lemmatization)
        df_disaster_tweet['final'] = df_disaster_tweet['text_lemmatized'].apply(' '.join)

        for column in ['text_lower', 'text_no_punct', 'text_tokenized', 'text_no_stop', 'text_lemmatized', 'final']:
            samples = df_disaster_tweet[column].head(5).tolist()
            samples_formatted = "\n".join(f"{index}. {item}"
                                          for index, item in enumerate(samples, start=1))
            logger.info(f"Sample from column '{column}':\n{samples_formatted}\n")

        data_disaster = df_disaster_tweet[df_disaster_tweet['target'] == 1]
        data_not_disaster = df_disaster_tweet[df_disaster_tweet['target'] == 0]

        logger.info('Data Disaster Shape: %s', data_disaster.shape)
        logger.info('Data Not Disaster Shape: %s', data_not_disaster.shape)
        logger.info('✅ Data preprocessing completed with success!')

        return df_disaster_tweet, data_disaster, data_not_disaster

    except Exception as e:
        print(f"❌ Error in text_preprocessing function: {e}")
        return None, None, None

def disaster_cloud(data_disaster, logger, run):
    """
    Generates and logs a WordCloud for disaster-related tweets.
    """
    wordcloud_disaster = WordCloud(max_words=500, random_state=100, background_color='white',
                                  collocations=True).generate(str((data_disaster['final'])))
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud_disaster, interpolation='bilinear')
    plt.title('WordCloud of the Disaster Tweets')
    plt.axis("off")
    disaster_path = 'wordcloud_disaster.png'
    plt.savefig(disaster_path)
    plt.close()

    logger.info("⏳ Uploading Figure 1 - WordCloud Disaster Tweets")
    run.log({"WordCloud Disaster Tweets": wandb.Image(disaster_path)})

def non_disaster_cloud(data_not_disaster, logger, run):
    """
    Generates and logs a WordCloud for non-disaster-related tweets.
    """
    wordcloud_not_disaster = WordCloud(max_words=500, random_state=100, background_color='white',
                                       collocations=True).generate(str((data_not_disaster['final'])))
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud_not_disaster, interpolation='bilinear')
    plt.title('WordCloud of the Non-Disaster Tweets')
    plt.axis("off")
    not_disaster_path = 'wordcloud_not_disaster.png'
    plt.savefig(not_disaster_path)
    plt.close()

    logger.info("⏳ Uploading Figure 2 - WordCloud Non-Disaster Tweets")
    run.log({"WordCloud Non-Disaster Tweets": wandb.Image(not_disaster_path)})

def create_preprocessing_artifact(df_disaster_tweet, logger):
    """
    Creates a Weights & Biases artifact for the preprocessed data.
    """
    try:
        preprocessing_artifact = wandb.Artifact('processed_data', type='Preprocessing',
                                 description='Preprocessing for Disaster-Related Tweets')
        processed_data_path = 'df_disaster_tweet_processed.csv'
        df_disaster_tweet.to_csv(processed_data_path, index=False)
        preprocessing_artifact.add_file(processed_data_path)
        logger.info('✅ Preprocessing artifact created with success!')
        return preprocessing_artifact

    except Exception as e:
        print(f"❌ Error in create_preprocessing_artifact function: {e}")
        return None

def preprocessing():
    """
    Main function for preprocessing
    """
    logger = setup_logging()

    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            try:
                install_libraries(logger)
                df_disaster_tweet, run = download_dataset_artifact(logger)
                if df_disaster_tweet is not None:
                    df_disaster_tweet, data_disaster, data_not_disaster = text_preprocessing(df_disaster_tweet, logger)
                    disaster_cloud(data_disaster, logger, run)
                    non_disaster_cloud(data_not_disaster, logger, run)
                    preprocessing_artifact = create_preprocessing_artifact(df_disaster_tweet, logger)
                    run.log_artifact(preprocessing_artifact)
                    run.finish()
            except Exception as e:
                logger.error(f"❌ Error during the execution of exploratory data analysis: %s", e)
        else:
            logger.error("❌ API key not found in the configuration file.")
    else:
        logger.error("❌ Unable to load the configuration.")

preprocessing()
