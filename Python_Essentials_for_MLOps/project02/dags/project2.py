'''
This module aims to create a data pipeline with
Apache Airflow to download poadcasts from
marketplace
'''

import json
import logging
import requests
import pendulum
import xmltodict
from typing import List, Dict
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

from airflow.decorators import dag, task
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000

def create_database() -> SqliteOperator:
    '''
    This function creates a SQLite database table to store information about the podcasts.

    Returns:
        SqliteOperator: An instance of SqliteOperator representing 
        the task to create the database table.
    '''
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
    create_database_task = SqliteOperator(
        task_id='create_table_sqlite',
        sql=create_table_sql,
        sqlite_conn_id="podcasts"
    )
    return create_database_task

@task()
def get_episodes() -> List[Dict[str, str]]:
    '''
    This task retrieves the latest episodes from the PODCAST_URL and returns them as a list.
    
    Returns:
        list: A list of dictionaries representing the latest podcast episodes.
            Each dictionary contains keys: "link", "title", "pubDate", and "description".
    '''
    try:
        response = requests.get(PODCAST_URL, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad requests

        feed = xmltodict.parse(response.text)
        episodes = feed["rss"]["channel"]["item"]

        # Parse episode information into a list of dictionaries
        parsed_episodes = []
        for episode in episodes:
            episode_data = {
                "link": episode.get("link", ""),
                "title": episode.get("title", ""),
                "pubDate": episode.get("pubDate", ""),
                "description": episode.get("description", "")
            }
            parsed_episodes.append(episode_data)

        num_episodes = len(parsed_episodes)
        logging.info("ℹ️ Found %s episodes.", num_episodes)
        return parsed_episodes

    except requests.exceptions.HTTPError as http_err:
        logging.error("HTTP error occurred: %s", http_err)
        raise
    except Exception as err:
        logging.error("An error occurred: %s", err)
        raise

@task()
def load_episodes(episodes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    '''
    This task loads new episodes into the SQLite database.
    
    Args:
        episodes (list): A list of dictionaries containing new episodes' information.
            Each dictionary should have keys: "link", "title", "pubDate", and "description".

    Returns:
        list: A list of dictionaries representing episodes successfully loaded into the database.
            Each dictionary contains keys: 
            "link", "title", "pubDate", "description", and "filename".
    '''
    try:
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []

        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append({
                    "link": episode["link"],
                    "title": episode["title"],
                    "pubDate": episode["pubDate"],
                    "description": episode["description"],
                    "filename": filename
                })

        hook.insert_rows(table='episodes', rows=new_episodes,
        target_fields=["link", "title", "published", "description", "filename"])
        logging.info("✅ Loaded %s new episodes into the database.", len(new_episodes))
        return new_episodes
    except Exception as err:
        logging.error("❌ An error occurred while loading episodes: %s", err)
        raise

@task()
def download_episodes(episodes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    '''
    This task loads new episodes into the SQLite database.
    
    Args:
        episodes (list): A list of dictionaries containing new episodes' information.
            Each dictionary should have keys: 
            "link", "title", "pubDate", and "description".

    Returns:
        list: A list of dictionaries representing episodes successfully loaded into the database.
            Each dictionary contains keys: 
            "link", "title", "pubDate", "description", and "filename".
    '''
    try:
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []

        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append({
                    "link": episode["link"],
                    "title": episode["title"],
                    "pubDate": episode["pubDate"],
                    "description": episode["description"],
                    "filename": filename
                })

        hook.insert_rows(table='episodes', rows=new_episodes,
                         target_fields=["link", "title", "published", "description", "filename"])
        logging.info("✅ Loaded %s new episodes into the database.", len(new_episodes))
        return new_episodes
    except Exception as err:
        logging.error("❌ An error occurred while loading episodes: %s", err)
        raise

@task()
def speech_to_text(audio_files: List[Dict[str, str]], new_episodes: List[Dict[str, str]]) -> int:
    '''
    This task transcribes audio files into text and updates the database with the transcripts.
    
    Args:
        audio_files (list): A list of dictionaries containing information about audio files.
            Each dictionary should have keys: "filename" (str) and "file_path" (str).
        new_episodes (list): A list of dictionaries representing 
        new episodes to be transcribed.
            Each dictionary should have keys: 
            "link" (str), "filename" (str), and other episode details.

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
                audio_file_path = [entry["file_path"] for entry in audio_files
                if entry["filename"] == filename][0]
                mp3 = AudioSegment.from_mp3(audio_file_path)
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
    '''
    DAG for processing podcast data pipeline.
    '''
    create_database_task = create_database()
    episodes = get_episodes()
    new_episodes = load_episodes(episodes)
    audio_files = download_episodes(episodes)
    transcribed_count = speech_to_text(audio_files, new_episodes)

    create_database_task >> episodes >> new_episodes >> audio_files >> transcribed_count

SUMMARY = podcast_summary3()
