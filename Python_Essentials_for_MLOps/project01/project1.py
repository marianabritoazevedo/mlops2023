"""
This module aims to create two systems: one to find the 5 most 
similar movies based on their title, and a recommendation system 
to find the 10 most similar movies based on their ratings.
"""

import re
import logging
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NoDataFiltered(Exception):
    '''
    Raised when there is no search when applying a filter 
    '''

def read_csv_file(file_path: str) -> pd.DataFrame:
    '''
    Reads a CSV file and returns its content as a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data 
        from the CSV file, or None if the file is not found.
    '''
    try:
        data = pd.read_csv(file_path)
        logging.info('✅ Data loaded successfully from file: %s', file_path)
        return data
    except FileNotFoundError:
        logging.error('❌ File not found at path: %s', file_path)
        return None

def display_dataframe(data: pd.DataFrame) -> None:
    '''
    Displays the first 5 rows of a Pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be displayed.

    Returns:
        None
    '''
    try:
        if isinstance(data, pd.DataFrame):
            display(data.head())
        else:
            raise ValueError('❌ Invalid input: Not a Pandas DataFrame')
    except ValueError as error:
        logging.error(error)


def column_types(data: pd.DataFrame) -> None:
    '''
    Displays the data types of columns in a Pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame for which column types are to be displayed.

    Returns:
        None
    '''
    try:
        if isinstance(data, pd.DataFrame):
            display(data.dtypes)
        else:
            raise ValueError('❌ Invalid input: Not a Pandas DataFrame')
    except ValueError as error:
        logging.error(error)

def clean_title(title: str) -> str:
    '''
    Removes special characters from a string, retaining only letters, numbers, and white spaces.

    Args:
        title (str): The input string containing special characters.

    Returns:
        str: The cleaned string without special characters.
    '''
    cleaned_title = re.sub(r"[^a-zA-Z0-9 ]", "", title)
    return cleaned_title

def search_similar_movies_title(title: str, data: pd.DataFrame, vectorizer, tfidf) -> pd.DataFrame:
    '''
    Returns movies most similar to the provided movie based on its title.

    Args:
        title (str): The movie title to search for.
        data (pd.DataFrame): The DataFrame containing movie data.
        vectorizer: The vectorizer used to transform movie titles into vectors.
        tfidf: The TF-IDF vectorizer used for movie title similarity calculation.

    Returns:
        pd.DataFrame: DataFrame containing movies most similar to the provided title.
    '''
    cleaned_title = clean_title(title)
    query_vec = vectorizer.transform([cleaned_title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = data.iloc[indices].iloc[::-1]
    return results

def filter_movie_by_id(movie_id: int, data: pd.DataFrame) -> pd.DataFrame:
    '''
    Filters a movie by its ID in the provided DataFrame.

    Args:
        movie_id (int): The ID of the movie to search for.
        data (pd.DataFrame): The DataFrame containing movie data.

    Returns:
        pd.DataFrame or None: DataFrame containing the filtered movie, or None if no movie is found.
    '''
    try:
        filtered_movie = data[data["movieId"] == movie_id]
        if len(filtered_movie) == 0:
            raise NoDataFiltered('❌ There isn\'t any movie with this ID.')
        return filtered_movie
    except NoDataFiltered as error:
        logging.error(error)
        return None

def search_similar_movies_ratings(movie_id: int, ratings: pd.DataFrame,
                                  movies: pd.DataFrame) -> pd.DataFrame:
    '''
    Searches for similar movies based on users ratings.

    Args:
        movie_id (int): The ID of the movie to search for similar movies.
        ratings (pd.DataFrame): DataFrame containing users' movie ratings.
        movies (pd.DataFrame): DataFrame containing movie information.

    Returns:
        pd.DataFrame: DataFrame containing top recommendations similar to the provided movie.
    '''
    # Find users who rated the movie with the specified ID and rated it higher than 4
    similar_users = ratings[(ratings["movieId"] == movie_id)
                            & (ratings["rating"] > 4)]["userId"].unique()

    # Filter the movie ratings made by similar users and a rating higher than 4
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users))
                                & (ratings["rating"] > 4)]["movieId"]

    # Calculate the recommendation percentage for each movie based on ratings from similar users
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # Filter movies with a recommendation percentage greater than 10%
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    # Filter all movie ratings made by similar users
    # and calculates its recommendation percentage
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index))
                        & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # Concatenate the recommendation percentages from similar users and all users
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    # Calculate the score as the ratio of recommendation
    # percentages from similar users and all users
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # Return the top 10 recommendations
    cols = ["score", "title", "genres"]
    results = rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[cols]
    return results

def main():
    '''
    Main function that serves as an entry point for the program execution.
    '''
    # Load movies data from CSV file
    movies = read_csv_file("files/movies.csv")
    display_dataframe(movies)

    # Clean movie titles and display the updated dataframe
    movies["clean_title"] = movies["title"].apply(clean_title)
    display_dataframe(movies)

    # Initialize TfidfVectorizer for movie titles
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])

    # Define widgets for user input
    movie_input = widgets.Text(value='Toy Story', description='Movie Title:', disabled=False)
    movie_list = widgets.Output()

    def on_type_title(data):
        '''
        Callback function called every time the movie_input changes. 
        Clears movie_list and searches for similar movies when the title 
        has more than 5 characters.
        '''
        with movie_list:
            movie_list.clear_output()
            title = data["new"]
            if len(title) > 5:
                display(search_similar_movies_title(title, movies, vectorizer, tfidf))
            else:
                logging.info('ℹ️ Enter at least 6 characters to search for similar movies!')

    movie_input.observe(on_type_title, names='value')
    display(movie_input, movie_list)

    # Load ratings data from CSV file and display it
    ratings = read_csv_file("files/ratings.csv")
    display_dataframe(ratings)
    column_types(ratings)

    # Filter movie by ID and display it
    movie_id = 89745
    movie = filter_movie_by_id(movie_id, movies)
    display(movie)

    # Find similar movies for the filtered movie and display recommendations
    similar_movies = search_similar_movies_ratings(movie_id, ratings, movies)
    display(similar_movies)

    # Define widgets for recommending movies based on user input
    movie_name_input = widgets.Text(value='Toy Story', description='Movie Title:', disabled=False)
    recommendation_list = widgets.Output()

    def on_type_rec(data):
        '''
        Callback function called every time the movie_name_input changes. 
        Clears recommendation_list and searches for movie recommendations 
        when the title has more than 5 characters.
        '''
        with recommendation_list:
            recommendation_list.clear_output()
            title = data["new"]
            if len(title) > 5:
                results = search_similar_movies_title(title, movies, vectorizer, tfidf)
                movie_id = results.iloc[0]["movieId"]
                display(search_similar_movies_ratings(movie_id, ratings, movies))
            else:
                logging.info('ℹ️ Enter at least 6 characters to search for similar movies!')

    movie_name_input.observe(on_type_rec, names='value')
    display(movie_name_input, recommendation_list)

if __name__== "__main__":
    main()
