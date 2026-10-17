#!/usr/bin/env python3

# Note: Don't change the next line.
from pyspark import SparkContext
import re

# Student ID:3334661 

# Note: Don't change the next line.
sc = SparkContext('local', 'myapp')


# Load the ratings data
ratings_rdd = sc.textFile('ratings.txt')

def parse_line(line):
    """
    Parse a line from the ratings file.
    Returns a tuple of (reviewer_id, movie_title, genres, year, rating).
    """
    parts = line.strip().split('\t')
    if len(parts) == 5:
        reviewer_id, movie_title, genres, year, rating = parts
        return (reviewer_id, clean_title(movie_title), genres, int(year), float(rating))
    return None

def clean_title(title):
    """
    Clean movie title by removing special characters and extra spaces.
    """
    title = re.sub(r"[^\w\s]", "", title)  # Keep only alphanumeric and spaces
    title = re.sub(r"\s+", " ", title).strip()  # Remove extra spaces
    return title

def filter_by_years_and_genres(movie, years, genres, all_years, all_genres):
    """
    Filter movies based on the specified years and genres.
    """
    _, _, movie_genres, year, _ = movie
    if not all_years and year not in years:
        return False
    movie_genres_list = movie_genres.split('|')
    if not all_genres:
        valid_genres = [genre for genre in movie_genres_list if genre in genres]
        return bool(valid_genres)
    return True

# Load years and genres
years = set()
with open('years.txt', 'r') as f:
    for line in f:
        for year in line.strip().split():
            years.add(int(year))
all_years = len(years) == 0  # If empty, consider all years

genres = set()
with open('genres.txt', 'r') as f:
    for line in f:
        for genre in line.strip().split():
            genres.add(genre)
all_genres = len(genres) == 0  # If empty, consider all genres


# Parse and filter the data
filtered_movies_rdd = (
    ratings_rdd.map(parse_line)
    .filter(lambda x: x is not None and filter_by_years_and_genres(x, years, genres, all_years, all_genres))
)

# Create (genre, (movie_title, rating)) pairs
genre_movie_rating_rdd = filtered_movies_rdd.flatMap(
    lambda movie: [(genre, (movie[1], movie[4])) for genre in movie[2].split('|') if all_genres or genre in genres]
)

# Create ((genre, movie_title), (rating, 1)) pairs and aggregate
genre_movie_total_rating_count_rdd = genre_movie_rating_rdd.map(
    lambda x: ((x[0], x[1][0]), (x[1][1], 1))
).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

# Calculate average rating
genre_movie_avg_rating_rdd = genre_movie_total_rating_count_rdd.mapValues(
    lambda x: (x[0] / x[1], x[1])  # (average_rating, count)
)

# Filter movies with at least 15 ratings
min_votes = 15
filtered_rdd = genre_movie_avg_rating_rdd.filter(lambda x: x[1][1] >= min_votes)

# Map to (genre, (movie_title, avg_rating))
genre_movie_avg_rating_mapped_rdd = filtered_rdd.map(
    lambda x: (x[0][0], (x[0][1], round(x[1][0], 4)))
)

# Group by genre and find the highest-rated movies (handling ties)
best_movies_by_genre_rdd = genre_movie_avg_rating_mapped_rdd.groupByKey().flatMap(
    lambda x: [(x[0], movie[0], movie[1]) for movie in x[1] if movie[1] == max(y[1] for y in x[1])]
)

# Note: Don't change the next line.
print('\n'.join(map(str, best_movies_by_genre_rdd.collect())), file=open('results.txt', 'w'))

# Note: Don't change the next line.
sc.stop()
