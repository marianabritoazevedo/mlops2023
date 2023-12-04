"""
This file aims to train disaster
tweets dataset with 4 different
models
"""

import json
import wandb
import logging
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import RMSprop
from codecarbon import EmissionsTracker

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

def download_artifacts(logger):
    """
    Download train_x, train_y, val_x, val_y
    and vocab artifacts from Wandb
    """
    try:
        # Start run in wandb
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="train")

        # Download train_y
        train_x_artifact = run.use_artifact('train_x:latest')
        train_x_path = train_x_artifact.file()
        train_x = joblib.load(train_x_path)

        # Download train_y
        train_y_artifact = run.use_artifact('train_y:latest')
        train_y_path = train_y_artifact.file()
        train_y = joblib.load(train_y_path)

        # Download val_x
        val_x_artifact = run.use_artifact('val_x:latest')
        val_x_path = val_x_artifact.file()
        val_x = joblib.load(val_x_path)

        # Download val_y
        val_y_artifact = run.use_artifact('val_y:latest')
        val_y_path = val_y_artifact.file()
        val_y = joblib.load(val_y_path)

        logger.info('✅ Artifacts downloaded with success!')
        return run, train_x, train_y, val_x, val_y
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return None, None, None, None, None

def set_parameters_and_layers(train_x):
    """
    Defines the parameters and layers (vectorizer
    and embedding) to build neural networks
    """
    max_tokens = 7500
    input_length = 128
    output_dim = 128
    vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                         output_mode='int',
                                                         standardize='lower_and_strip_punctuation',
                                                         output_sequence_length=input_length)
    vectorizer_layer.adapt(train_x)
    embedding_layer = Embedding(input_dim=max_tokens, 
                                output_dim=output_dim, 
                                input_length=input_length)
    return vectorizer_layer, embedding_layer

def model1(vectorizer_layer, embedding_layer):
    """
    Define the first model, a shallow neural
    network
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer_layer)
    model.add(embedding_layer),
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def train_model1(model, X_train, y_train, x_val, y_val, logger, run):
    """
    Train the first model
    """
    opt = tf.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    logger.info('⏳ Starting model 1 training')

    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.fit(X_train, y_train, epochs=10, verbose=2, validation_data=(x_val,y_val),
              callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
    
    tracker.stop()
    logger.info('✅ Model 1 training finished with success!')
    plot_loss_and_acc(history, 10, "Model1", run)
    energy_and_co2(tracker, logger)

    model_artifact = wandb.Artifact('modelo1', type='model')
    model.save(wandb.run.dir + '/model_tf', save_format='tf')
    model_artifact.add_dir(wandb.run.dir + '/model_tf')
    run.log_artifact(model_artifact)
    logger.info('✅ Model 1 saved with success!')

def model2(vectorizer_layer, embedding_layer):
    """
    Define the second model, a multi-layer
    neural network
    """
    model_regularized = tf.keras.models.Sequential()
    model_regularized.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_regularized.add(vectorizer_layer)
    model_regularized.add(embedding_layer)
    model_regularized.add(tf.keras.layers.GlobalAveragePooling1D())
    model_regularized.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=L1(0.0005)))
    model_regularized.add(tf.keras.layers.Dropout(0.6))
    model_regularized.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L1L2(0.0005)))
    model_regularized.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2(0.0005)))
    model_regularized.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2(0.0005)))
    model_regularized.add(tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=L2(0.0005)))
    model_regularized.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model_regularized

def train_model2(model, X_train, y_train, x_val, y_val, logger, run):
    """
    Train the second model
    """
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    logger.info('⏳ Starting model 2 training')

    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.fit(X_train, y_train, epochs=10, verbose=2, validation_data=(x_val,y_val),
              callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
    
    tracker.stop()
    logger.info('✅ Model 2 training finished with success!')
    plot_loss_and_acc(history, 10, "Model2", run)
    energy_and_co2(tracker, logger)

def model3(vectorizer_layer, embedding_layer):
    """
    Define the third model, a Multilayer Bidirectional
    LSTM neural network
    """
    ml_bi_lstm = Sequential()
    ml_bi_lstm.add(Input(shape=(1,), dtype=tf.string))
    ml_bi_lstm.add(vectorizer_layer)
    ml_bi_lstm.add(embedding_layer)
    ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
    ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
    ml_bi_lstm.add(Bidirectional(LSTM(64)))
    ml_bi_lstm.add(Dense(64, activation='elu', kernel_regularizer=L1L2(0.0001)))
    ml_bi_lstm.add(Dense(32, activation='elu', kernel_regularizer=L2(0.0001)))
    ml_bi_lstm.add(Dense(8, activation='elu', kernel_regularizer=L2(0.0005)))
    ml_bi_lstm.add(Dense(8, activation='elu'))
    ml_bi_lstm.add(Dense(4, activation='elu'))
    ml_bi_lstm.add(Dense(1, activation='sigmoid'))
    return ml_bi_lstm

def train_model3(model, X_train, y_train, x_val, y_val, logger, run):
    """
    Train the third model
    """
    opt = RMSprop(learning_rate=0.0001, rho=0.8, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    logger.info('⏳ Starting model 3 training')

    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.fit(X_train, y_train, epochs=3, validation_data=(x_val,y_val),
          callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
    
    tracker.stop()
    logger.info('✅ Model 3 training finished with success!')
    plot_loss_and_acc(history, 3, "Model3", run)
    energy_and_co2(tracker, logger)

def model4(train_x, train_y, val_x, val_y):
    """
    Define the fourth model, a Transformer
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Tokenize the text data
    train_encodings = tokenizer(list(train_x), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_x), truncation=True, padding=True)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        tf.constant(train_y.values, dtype=tf.int32)
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        tf.constant(val_y.values, dtype=tf.int32)
    ))

    train_dataset = train_dataset.batch(16)
    val_dataset = val_dataset.batch(16)
    return tokenizer, model, train_dataset, val_dataset

def train_model4(tokenizer, model_before, train_dataset, val_dataset, logger, run):
    """
    Train the fourth model
    """
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    logger.info('⏳ Starting model 4 training')
 
    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.fit(train_dataset, epochs=2, validation_data=val_dataset)

    tracker.stop()
    logger.info('✅ Model 4 training finished with success!')
    plot_loss_and_acc(history, 2, "Model4", run)
    energy_and_co2(tracker, logger)

def plot_loss_and_acc(history, epochs, name_model, run):
    """
    Plot loss and accuracy for each epoch
    while training the neural network
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1,1,figsize=(10,8))

    ax.plot(np.arange(0, epochs), history.history["loss"], label="train_loss",linestyle='--')
    ax.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss",linestyle='--')
    ax.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    ax.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    ax.set_title(f"Training Loss and Accuracy {name_model}")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()
    plt.show()

    image_path = f'{name_model}_train_graph.png'
    fig.savefig(image_path)
    plt.close()
    run.log({f"Loss and acc {name_model}": wandb.Image(image_path)})

def energy_and_co2(tracker, logger):
    """
    Show information about energy consumed
    and CO2 emission
    """
    logger.info("[INFO ⚡] {} kWh of electricity used since the begining".format(tracker.final_emissions_data.energy_consumed))
    logger.info("[INFO ⚡] Energy consumed for RAM: {} kWh".format(tracker.final_emissions_data.ram_energy))
    logger.info("[INFO ⚡] Energy consumed for all GPU: {} kWh".format(tracker.final_emissions_data.gpu_energy))
    logger.info("[INFO ⚡] Energy consumed for all CPU: {} kWh".format(tracker.final_emissions_data.cpu_energy))
    logger.info("[INFO 🍃] CO2 emission {}(in Kg)".format(tracker.final_emissions_data.emissions))

def train():
    """
    Main function for vocabulary
    """
    logger = setup_logging()
    file_config = load_config(logger)
    if file_config:
        file_api_key = file_config.get('wandb', {}).get('api_key')
        if file_api_key:
            run, train_x, train_y, val_x, val_y = download_artifacts(logger)
            vectorizer_layer, embedding_layer = set_parameters_and_layers(train_x)
            # First model
            model_1 = model1(vectorizer_layer, embedding_layer)
            train_model1(model_1, train_x, train_y, val_x, val_y, logger, run)
            # Second model
            model_2 = model2(vectorizer_layer, embedding_layer)
            train_model2(model_2, train_x, train_y, val_x, val_y, logger, run)
            # Third model
            model_3 = model3(vectorizer_layer, embedding_layer)
            train_model3(model_3, train_x, train_y, val_x, val_y, logger, run)
            # Fourth model
            tokenizer, model, train_dataset, val_dataset = model4(train_x, train_y, val_x, val_y)
            train_model4(tokenizer, model, train_dataset, val_dataset, logger, run)
            run.finish()
        else:
            logger.error("❌ API key not found in the config file.")
    else:
        logger.error("❌ Unable to load the configuration.")

train()