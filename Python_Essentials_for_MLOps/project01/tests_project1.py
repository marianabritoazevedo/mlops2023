from project1 import filter_movie_by_id, read_file

def test_verify_columns_dataset_movies():
    movies = read_file('files/movies.csv')
    columns = list(movies.columns)
    assert columns == ['movieId', 'title', 'genres']

def test_verify_columns_dataset_ratings():
    ratings = read_file('files/ratings.csv')
    columns = list(ratings.columns)
    assert columns == ['userId', 'movieId', 'rating', 'timestamp']

def test_filter_movie_by_id():
    movies = read_file('files/movies.csv')
    id_movie = 89745
    movie = filter_movie_by_id(id_movie, movies)
    assert movie.title.values[0] == 'Avengers, The (2012)'

