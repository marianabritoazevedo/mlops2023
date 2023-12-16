import mlflow
import logging
import os
import pandas as pd
import pytest
from mlflow import MlflowClient
import subprocess

# Set up logging
def setup_logging() -> tuple:
    logger = logging.getLogger('data_checks')
    logger.setLevel(logging.INFO)
    log_file = "./logs/data_checks.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_file

logger, log_file = setup_logging()


def download_artifact_preprocessing() -> None:
    """
    Downloads the preprocessed dataset artifact from MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    client = MlflowClient()

    # Name of the experiment
    experiment_name = "TwitterSentimentAnalysis"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise Exception(f"The experiment '{experiment_name}' does not exist.")

    # List all runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"])
    
    # Filter runs by the name "preprocessing"
    run_id = None
    for run in runs:
        if run.data.tags.get('mlflow.runName') == "preprocessing":
            run_id = run.info.run_id
            break

    if run_id is not None:
        data_dir = "./processed_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the artifact
        client.download_artifacts(run_id, "preprocessed_twitter_sentiment.csv", data_dir)
    else:
        raise Exception("No runs found with the name 'preprocessing' for experiment '{}'.".format(experiment_name))

# Fixture to prepare the data
@pytest.fixture
def processed_data() -> pd.DataFrame:
    """
    Loads the preprocessed dataset.
    """
    download_artifact_preprocessing()  # This function downloads the preprocessed dataset
    dataset_path = "./processed_data/preprocessed_twitter_sentiment.csv"  # Updated path to the downloaded dataset
    df = pd.read_csv(dataset_path)
    return df

def test_processed_data_columns(processed_data) -> None:
    """
    Tests the columns of the processed DataFrame.    
    """
    expected_columns = ['clean_text', 'sentiment']
    assert all(column in processed_data.columns for column in expected_columns), "The columns of the processed DataFrame are incorrect."

def test_processed_data_content(processed_data) -> bool:
    """
    Tests the content of the processed DataFrame.
    """
    assert not processed_data['clean_text'].isnull().any(), "The 'clean_text' column should not contain null values."
    assert processed_data['sentiment'].isin([-1, 0, 1, 2]).all(), "Values in 'sentiment' are not as expected."


def run_tests_and_log() -> None:
    """
    Runs the tests and logs the results.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    with mlflow.start_run(run_name="data_checks"):
        logger.info("🚀 Starting data checks...")

        # Run the tests with pytest
        test_output = subprocess.run(['pytest', '-v', os.path.abspath(__file__)], capture_output=True, text=True)

        logger.info("Test Output:\n" + test_output.stdout)
        if test_output.stderr:
            logger.error("Test Errors:\n" + test_output.stderr)
        # Log the log file as an artifact in MLflow
        mlflow.log_artifact(log_file)

if __name__ == "__main__":
    run_tests_and_log()