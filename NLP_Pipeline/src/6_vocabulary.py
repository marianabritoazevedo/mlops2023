"""
This file aims to create the vocabulary
related to disaster tweets train data
"""

from collections import Counter
import logging
import json
import joblib
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

def download_train_artifact(logger):
    """
    Download the train_x artifact from Weights & Biases 
    and transforms into a list
    """
    try:
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="vocabulary")
        train_x_artifact = run.use_artifact('train_x:latest')
        train_x_path = train_x_artifact.file()
        train_x = joblib.load(train_x_path)
        logger.info('✅ Train_x artifact loaded with success!')
        texts = list(train_x)
        return run, texts
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return None, None

def organize_tokens(doc):
    """
    Organize tokens of each doc to 
    create vocabulary
    """
    tokens = doc.split()
    return tokens

def add_docs_to_vocab(texts, vocab):
    """
    Update the vocabulary with clean tokens
    """
    for doc in texts:
        tokens = organize_tokens(doc)
        vocab.update(tokens)

def save_list(lines, filename):
    """
    Save the list with the vocabulary from
    train data
    """
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    file_vocab = open(filename, 'w')
    file_vocab.write(data)
    file_vocab.close()

def vocabulary_artifact(logger, run, texts):
    """
    Create the vocabulary and add as an artifact
    in Wandb
    """
    # Define vocab
    vocab = Counter()

    # Add all docs to vocab
    add_docs_to_vocab(texts, vocab)

    # Initial vocab
    size_vocab = len(vocab)
    wandb.log({'Initial vocab size': size_vocab})
    logger.info('Initial vocab size: %s', size_vocab)

    # Filtered vocab (tokens with at least 2 occurrences)
    min_occurrence = 2
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    size_vocab_filter = len(tokens)
    wandb.log({'Filtered vocab size': size_vocab_filter})
    logger.info('Filtered vocab size: %s', size_vocab_filter)

    # Save tokens to a vocabulary file
    save_list(tokens, 'vocabulary.txt')

    # Create a new artifact for the vocabulary
    vocab_artifact = wandb.Artifact(name='vocab', type='Vocab',
                                    description='Vocabulary from training data')
    vocab_artifact.add_file('vocabulary.txt')
    run.log_artifact(vocab_artifact)

def vocabulary():
    """
    Main function for vocabulary
    """
    logger = setup_logging()
    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            run, texts = download_train_artifact(logger)
            vocabulary_artifact(logger, run, texts)
            run.finish()
        else:
            logger.error("❌ API key not found in the config file.")
    else:
        logger.error("❌ Unable to load the configuration.")

vocabulary()
