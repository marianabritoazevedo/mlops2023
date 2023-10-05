import os
import pytest
import sqlite3
import requests
from unittest.mock import patch, Mock
from functions import create_database, get_episodes, load_episodes, download_episodes, speech_to_text

PODCAST_URL = 'https://www.marketplace.org/feed/podcast/marketplace/'


def test_create_database():
    '''
    Test the create_database function.
    This test checks if the function runs without errors.
    '''
    create_db_task = create_database()
    assert create_db_task.task_id == 'create_table_sqlite'

def test_get_episodes_integration():
    '''
    Integration test for the get_episodes function.
    This test makes a real request to the test endpoint, 
    ensuring there are no errors in the response.
    It then checks if the response status code is 200 (OK) 
    and if the number of episodes returned is greater than 0.
    '''
    # Make a real request to the test endpoint
    response = requests.get(PODCAST_URL)
    response.raise_for_status()  # Ensure there are no errors in the response

    # Get episodes using the function from your code
    episodes = get_episodes()

    # Verify if the response status code is 200 (OK)
    assert response.status_code == 200

    # Verify if the number of episodes returned is greater than 0
    assert len(episodes) > 0
