"""
This file conducts an Exploratory Data Analysis 
(EDA) on disaster-related tweets.
"""

import json
import logging
import subprocess
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
            logger.info('✅ File loaded successfully!')
        return config
    except FileNotFoundError:
        logger.error("❌ Configuration file not found.")
        return None
    except json.JSONDecodeError:
        logger.error("❌ Error decoding JSON in the configuration file.")
        return None

def install_libraries(logger):
    """
    Install the required libraries to run fetch_data.py.
    """
    try:
        subprocess.run(['pip', 'install', 'wandb'], check=True)
        logger.info('✅ WandB library installed successfully!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error installing libraries: {e}")

def download_eda_artifact(logger):
    """
    Initialize a wandb run, download an artifact, and load a dataset for EDA.
    """
    try:
        # Initialize wandb run
        run = wandb.init(project='disaster_tweet_classification', save_code=True, job_type="eda")

        # Get the artifact
        artifact = run.use_artifact('dataset:v0')

        # Download the content of the artifact to the local directory
        artifact_dir = artifact.download()

        # Path to dataset
        data_path = artifact_dir + '/train.csv'

        # Load dataset
        df_disaster_tweet = pd.read_csv(data_path)

        # Create an artifact for EDA
        eda_artifact = wandb.Artifact('eda', type='EDA',
                                       description='EDA for Disaster-Related Tweets')
        logger.info('✅ EDA artifact created with success!')

        return df_disaster_tweet, eda_artifact, run
    except Exception as e:
        logger.error(f"❌ Error during artifact download: {e}")
        return None, None, None

def general_info_dataset(logger, df_disaster_tweet, eda_artifact, run):
    """
    Display general information about the dataset to wandb.
    """
    try:
        # Log the shape of the DataFrame
        run.log({'DataFrame Shape': df_disaster_tweet.shape})
        logger.info('DataFrame Shape: %s', df_disaster_tweet.shape)

        # Convert the DataFrame to a dictionary and log it
        run.log({"Head": df_disaster_tweet.head().to_dict('records')})
        run.log({"Tail": df_disaster_tweet.tail().to_dict('records')})

        # Create tables in Wandb with the columns of the dataset, its head and its tail

        columns_df = pd.DataFrame({'Column': df_disaster_tweet.columns})
        run.log({'Columns Table': wandb.Table(dataframe=columns_df)})
        run.log({'Head Table': wandb.Table(dataframe=df_disaster_tweet.head())})
        run.log({'Tail Table': wandb.Table(dataframe=df_disaster_tweet.tail())})

        # Log unique value counts
        unique_counts = {
            'Keyword Unique': df_disaster_tweet['keyword'].nunique(),
            'Location Unique': df_disaster_tweet['location'].nunique(),
            'Text Unique': df_disaster_tweet['text'].nunique(),
            'ID Unique': df_disaster_tweet['id'].nunique(),
            'Target Unique': df_disaster_tweet['target'].nunique()
        }
        for k, v in unique_counts.items():
            run.log({k: v})
            logger.info('%s: %s', k, v)
        return eda_artifact
    except Exception as e:
        logger.error(f"❌ Error logging general dataset information: {e}")
        return None

def create_bar_graph(logger, df_disaster_tweet, run):
    """
    Create a bar graph and log it to wandb.
    """
    try:
        # Drop unnecessary columns
        df_disaster_tweet = df_disaster_tweet.drop(['id', 'keyword', 'location'], axis=1)

        # Log target counts and proportions
        target_counts = df_disaster_tweet['target'].value_counts().reset_index()
        target_proportion = df_disaster_tweet['target'].value_counts(normalize=True).reset_index()

        # Converting the result to a dictionary and logging it
        run.log({"Target Counts": target_counts.to_dict('records')})
        run.log({"Target Proportion": target_proportion.to_dict('records')})

        # Create tables in Wandb with Target Counts and Target Proportion
        run.log({'Target Counts Table': wandb.Table(dataframe=target_counts)})
        run.log({'Target Proportion Table': wandb.Table(dataframe=target_proportion)})

        # Convert the data to a format compatible with wandb.Table
        data = [[label, count] for label, count in
                df_disaster_tweet['target'].value_counts().items()]
        table = wandb.Table(data=data, columns=["label", "count"])

        # Use the wandb plotting function to create the count plot
        wandb.log({"Target Count Plot":
                    wandb.plot.bar(table, "label", "count", title="Distribution of Target")})

    except Exception as e:
        logger.error(f"❌ Error creating the bar graph: {e}")

def eda():
    """
    Main function for exploratory data analysis.
    """
    logger = setup_logging()

    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            try:
                install_libraries(logger)
                df_disaster_tweet, eda_artifact, run = download_eda_artifact(logger)
                if df_disaster_tweet is not None:
                    eda_artifact = general_info_dataset(logger, df_disaster_tweet,
                    eda_artifact, run)
                    create_bar_graph(logger, df_disaster_tweet, run)
                    run.log_artifact(eda_artifact)
                    run.finish()
            except Exception as e:
                logger.error(f"❌ Error during the execution of exploratory data analysis: %s", e)
        else:
            logger.error("❌ API key not found in the configuration file.")
    else:
        logger.error("❌ Unable to load the configuration.")

eda()
