import os
import json
import logging
import requests
import pendulum
import xmltodict
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

from airflow.decorators import dag, task
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000

def create_database():
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
    """
    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=create_table_sql,
        sqlite_conn_id="podcasts"
    )
    return create_database

@task()
def get_episodes():
    '''
    This task retrieves the latest episodes from
    the PODCAST_URL and returns them as a list.
    
    Returns:
        list: List of latest podcast episodes.
    '''
    try:
        response = requests.get(PODCAST_URL, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad requests

        feed = xmltodict.parse(response.text)
        episodes = feed["rss"]["channel"]["item"]
        num_episodes = len(episodes)
        logging.info("ℹ️ Found %s episodes.", num_episodes)
        return episodes

    except requests.exceptions.HTTPError as http_err:
        logging.error("HTTP error occurred: %s", http_err)
        raise
    except Exception as err:
        logging.error("An error occurred: %s", err)
        raise

@task()
def load_episodes(episodes):
    '''
    This task loads new episodes into the SQLite database.
    
    Args:
        episodes (list): List of new episodes to be loaded.
        
    Returns:
        list: List of episodes successfully loaded into the database.
    '''
    try:
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []

        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"], episode["title"],
                episode["pubDate"], episode["description"], filename])

        hook.insert_rows(table='episodes', rows=new_episodes,
        target_fields=["link", "title", "published", "description", "filename"])
        logging.info("✅ Loaded %s new episodes into the database.", len(new_episodes))
        return new_episodes
    except Exception as err:
        logging.error("❌ An error occurred while loading episodes: %s", err)
        raise

@task()
def download_episodes(episodes):
    '''
    This task loads new episodes into the SQLite database.
    
    Args:
        episodes (list): List of new episodes to be loaded.
        
    Returns:
        list: List of episodes successfully loaded into the database.
    '''
    try:
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []

        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"], episode["title"],
                episode["pubDate"], episode["description"], filename])

        hook.insert_rows(table='episodes', rows=new_episodes,
        target_fields=["link", "title", "published", "description", "filename"])
        logging.info("✅ Loaded %s new episodes into the database.", len(new_episodes))
        return new_episodes
    except Exception as err:
        logging.error("❌ An error occurred while loading episodes: %s", err)
        raise

@task()
def speech_to_text(audio_files, new_episodes):
    '''
    This task transcribes audio files into text and updates the database with the transcripts.
    
    Args:
        audio_files (list): List of audio file information (e.g., file paths).
        new_episodes (list): List of new episodes to be transcribed.
        
    Returns:
        int: Number of episodes successfully transcribed and updated in the database.
    '''
    try:
        hook = SqliteHook(sqlite_conn_id="podcasts")
        transcribed_count = 0

        model = Model(model_name="vosk-model-en-us-0.22-lgraph")
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        for episode_info in new_episodes:
            episode_link = episode_info["link"]
            filename = episode_info["filename"]

            if any(entry["filename"] == filename for entry in audio_files):
                print(f"Transcribing {filename}")
                filepath = os.path.join(EPISODE_FOLDER, filename)
                mp3 = AudioSegment.from_mp3(filepath)
                mp3 = mp3.set_channels(1)
                mp3 = mp3.set_frame_rate(FRAME_RATE)

                step = 20000
                transcript = ""
                for i in range(0, len(mp3), step):
                    segment = mp3[i:i+step]
                    rec.AcceptWaveform(segment.raw_data)
                    result = rec.Result()
                    text = json.loads(result)["text"]
                    transcript += text

                # Update the database with the transcript
                hook.run("UPDATE episodes SET transcript=? WHERE link=?",
                (transcript, episode_link))
                transcribed_count += 1
        logging.info("✅ Transcribed %s episodes.", transcribed_count)
        return transcribed_count
    except Exception as err:
        logging.error("❌ An error occurred while transcribing episodes: %s", err)
        raise

@dag(
    dag_id='podcast_summary3',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary3():
    create_database_task = create_database()
    episodes = get_episodes()
    new_episodes = load_episodes(episodes)
    audio_files = download_episodes(episodes)
    transcribed_count = speech_to_text(audio_files, new_episodes)

    create_database_task >> episodes >> new_episodes >> audio_files >> transcribed_count

SUMMARY = podcast_summary3()