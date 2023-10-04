import pytest
from project1 import *

# Fixture to load movies data for testing
@pytest.fixture
def movies_data():
    return read_csv_file("files/movies.csv")

# Fixture to load ratings data for testing
@pytest.fixture
def ratings_data():
    return read_csv_file("files/ratings.csv")

def test_verify_columns_dataset_movies(movies_data):
    columns = list(movies_data.columns)
    assert columns == ['movieId', 'title', 'genres']

def test_verify_columns_dataset_ratings(ratings_data):
    columns = list(ratings_data.columns)
    assert columns == ['userId', 'movieId', 'rating', 'timestamp']

def test_clean_title():
    assert clean_title("The Dark Knight (2008)") == "The Dark Knight 2008"

def test_filter_movie_by_id(movies_data):
    id_movie = 89745
    movie = filter_movie_by_id(id_movie, movies_data)
    assert movie.title.values[0] == 'Avengers, The (2012)'

def test_search_similar_movies_ratings(movies_data, ratings_data):
    similar_movies = search_similar_movies_ratings(1, ratings_data, movies_data)
    assert len(similar_movies) == 10  # Assuming there are 10 similar movies for the given movie ID

# Parameterized test to check if movie titles are cleaned properly
@pytest.mark.parametrize("input_title, expected_output", [("Inception (2010)", "Inception 2010"),
                                                         ("Interstellar (2014)", "Interstellar 2014")])
def test_clean_title_parameterized(input_title, expected_output):
    assert clean_title(input_title) == expected_output