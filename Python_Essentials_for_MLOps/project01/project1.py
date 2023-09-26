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

def read_file(link):
    '''
    This function aims to read a csv file
    '''
    try:
        data = pd.read_csv(link)
        logging.info('✅ Data loaded successfully!')
        return data
    except FileNotFoundError:
        logging.error('❌ File not found.')
        return None

def show_file(data):
    '''
    This function aims to show the first 5 lines of a Pandas dataframe
    '''
    try:
        display(data.head())
    except AttributeError:
        logging.error('❌ It was not possible to show this file.')

def show_types(data):
    '''
    This function aims to show the types of the columns of a dataframe
    '''
    try:
        display(data.dtypes)
    except AttributeError:
        logging.error('❌ It was not possible to show the types of this file.')

def clean_title(title):
    '''
    This function removes special characters from a string, 
    retaining only letters, numbers, and white spaces.
    '''
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

def search_similar_movies_title(title, data, vectorizer, tfidf):
    '''This function returns the movies most similar to 
    the provided movie based on its title. '''
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = data.iloc[indices].iloc[::-1]
    return results

def filter_movie_by_id(movie_id, data):
    '''
    This function searchs a movie by its id
    '''
    try:
        movie = data[data["movieId"] == movie_id]
        if len(movie) == 0:
            raise NoDataFiltered('❌ There isn\'t any movie with this ID.')
        return movie
    except NoDataFiltered as error:
        print(error)
        return None

def search_similar_movies_ratings(movie_id, ratings, movies):
    '''
    This function searches for similar movies based on users ratings
    '''
    # Find users who rated the movie with the specified ID and rated it higher than 4
    query=(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)
    similar_users = ratings[query]["userId"].unique()

    # Filter the movie ratings made by similar users and a rating higher than 4
    query=(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)
    similar_user_recs = ratings[query]["movieId"]

    # Calculate the recommendation percentage for each movie based on ratings from similar users
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # Filter movies with a recommendation percentage greater than 10%
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    # Filter all movie ratings made by similar users and calculates its recommendation percentage
    query=(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)
    all_users = ratings[query]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # Concatenate the recommendation percentages from similar users and all users
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    # Calculate the score as the ratio of recommendation percentages from
    # similar users and all users
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # Return the top 10 recommendations
    cols=["score", "title", "genres"]
    results = rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[cols]
    return results

def main():
    '''
    This is the function executed when we run the program
    '''
    # List of movies
    movies = read_file("files/movies.csv")
    show_file(movies)

    # Display movies with clean title
    movies["clean_title"] = movies["title"].apply(clean_title)
    show_file(movies)

    # Creates an TfidfVectorizer instance
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    # Numeric representation of movie title based on words frequency and bigrams
    tfidf = vectorizer.fit_transform(movies["clean_title"])

    # Defines the movie input to search to similar movies
    movie_input = widgets.Text(
        value='Toy Story',
        description='Movie Title:',
        disabled=False
    )

    # List with 5 most similar movies
    movie_list = widgets.Output()

    def on_type_title(data):
        '''
        This function is a callback called every time the movie_input 
        changes. When this happens, the movie_list will be cleared, and 
        if the title has more than 5 characters, it will search the 
        most similar movies according to its title
        '''
        with movie_list:
            movie_list.clear_output()
            title = data["new"]
            if len(title) > 5:
                display(search_similar_movies_title(title, movies, vectorizer, tfidf))
            else:
                logging.info('ℹ️ Enter at least 6 characters to search for similar movies!')

    # Automatically calls on_type_title when movie_input changes
    movie_input.observe(on_type_title, names='value')
    # Display the movie and its similar ones
    display(movie_input, movie_list)

    # List of movies ratings
    ratings = read_file("files/ratings.csv")
    show_file(ratings)
    show_types(ratings)

    # Filter movie and show its recommendations
    movie_id = 89745
    movie = filter_movie_by_id(movie_id, movies)
    display(movie)
    similar_movies = search_similar_movies_ratings(movie_id, ratings, movies)
    display(similar_movies)

    # Defines the movie input to search to similar movies
    movie_name_input = widgets.Text(
        value='Toy Story',
        description='Movie Title:',
        disabled=False
    )

    # List with 10 movies to recommend
    recommendation_list = widgets.Output()

    def on_type_rec(data):
        '''
        This function is a callback called every time the movie_input 
        changes. When this happens, the recommendation_list will be 
        cleared, and if the title has more than 5 characters, it will 
        search 10 movies to recommend
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
