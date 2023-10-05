import pytest
from project1 import *

# Fixture to load movies data for testing
@pytest.fixture
def movies_data():
    '''
    Fixture to load movies data for testing.
    Returns:
        pd.DataFrame: Movies data loaded from the CSV file.
    '''
    return read_csv_file("files/movies.csv")

# Fixture to load ratings data for testing
@pytest.fixture
def ratings_data():
    '''
    Fixture to load ratings data for testing.
    Returns:
        pd.DataFrame: Ratings data loaded from the CSV file.
    '''
    return read_csv_file("files/ratings.csv")

def test_verify_columns_dataset_movies(movies_data):
    '''
    Test to verify columns in the movies dataset.
    It checks if the columns match the expected columns.
    Args:
        movies_data (pd.DataFrame): Movies dataset for testing.
    '''
    columns = list(movies_data.columns)
    assert columns == ['movieId', 'title', 'genres']

def test_verify_columns_dataset_ratings(ratings_data):
    '''
    Test to verify columns in the ratings dataset.
    It checks if the columns match the expected columns.
    Args:
        ratings_data (pd.DataFrame): Ratings dataset for testing.
    '''
    columns = list(ratings_data.columns)
    assert columns == ['userId', 'movieId', 'rating', 'timestamp']

def test_clean_title():
    '''
    Test for cleaning movie titles.
    It checks if the clean_title function correctly removes the release year from movie titles.
    '''
    assert clean_title("The Dark Knight (2008)") == "The Dark Knight 2008"

def test_filter_movie_by_id(movies_data):
    '''
    Test to filter movies by ID.
    It checks if the filter_movie_by_id function correctly filters movies by the given ID.
    Args:
        movies_data (pd.DataFrame): Movies dataset for testing.
    '''
    id_movie = 89745
    movie = filter_movie_by_id(id_movie, movies_data)
    assert movie.title.values[0] == 'Avengers, The (2012)'

def test_search_similar_movies_ratings(movies_data, ratings_data):
    '''
    Test to search for similar movies based on ratings.
    It checks if the search_similar_movies_ratings function returns the expected number of similar movies.
    Args:
        movies_data (pd.DataFrame): Movies dataset for testing.
        ratings_data (pd.DataFrame): Ratings dataset for testing.
    '''
    similar_movies = search_similar_movies_ratings(1, ratings_data, movies_data)
    assert len(similar_movies) == 10  # Assuming there are 10 similar movies for the given movie ID

# Parameterized test to check if movie titles are cleaned properly
@pytest.mark.parametrize("input_title, expected_output", [("Inception (2010)", "Inception 2010"),
                                                         ("Interstellar (2014)", "Interstellar 2014")])
def test_clean_title_parameterized(input_title, expected_output):
    '''
    Parameterized test to check if movie titles are cleaned properly.
    It uses input_title as the input and expected_output as the expected cleaned title.
    Args:
        input_title (str): Input movie title for testing.
        expected_output (str): Expected cleaned movie title.
    '''
    assert clean_title(input_title) == expected_output