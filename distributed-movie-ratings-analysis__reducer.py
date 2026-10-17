#!/usr/bin/env python3
# reducer.py
# Student ID:3334661
import sys
import logging
from collections import defaultdict

# Set up logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

MIN_VOTES = 15  # Minimum number of votes required for a movie to be considered

def main():
    # Dictionary to store aggregated ratings and counts for each (genre, movie_title) pair
    genre_movie_ratings = defaultdict(lambda: [0.0, 0])  # { (genre, movie_title) : [sum_ratings, count] }
    
    for line in sys.stdin:
        fields = line.strip().split('\t')
        if len(fields) < 4:
            logging.error(f"Malformed line skipped: {line.strip()}")
            continue  # Skip malformed lines
        
        genre, movie_title, total_rating, total_count = fields
        
        try:
            total_rating = float(total_rating)  # Convert total_rating to float
            total_count = int(total_count)      # Convert total_count to int
        except ValueError:
            logging.error(f"Invalid data skipped: {line.strip()}")
            continue  # Skip lines with invalid data
        
        # Re-aggregate ratings and counts for this (genre, movie_title) pair
        genre_movie_ratings[(genre, movie_title)][0] += total_rating  # Sum up ratings
        genre_movie_ratings[(genre, movie_title)][1] += total_count   # Sum up counts
    
    # Dictionary to store the best movie(s) for each genre
    genre_best_movies = defaultdict(lambda: (None, 0.0, 0))  # { genre : (best_movie, best_avg, total_count) }
    
    # Calculate the average rating for each movie-genre pair and find the best movie(s) for each genre
    for (genre, movie_title), (total_rating, total_count) in genre_movie_ratings.items():
        if total_count >= MIN_VOTES:
            avg_rating = total_rating / total_count
            best_movie, best_avg, _ = genre_best_movies[genre]
            if best_movie is None or avg_rating > best_avg:
                genre_best_movies[genre] = (movie_title, avg_rating, total_count)
            elif avg_rating == best_avg:
                best_movie, best_avg, best_count = genre_best_movies[genre]
                # Append the new movie title to the list
                if isinstance(best_movie, list):
                    best_movie.append(movie_title)
                else:
                    genre_best_movies[genre] = ([best_movie, movie_title], avg_rating, total_count)
    
    # Output the best movie(s) for each genre
    for genre, (movie_title, avg_rating, total_count) in genre_best_movies.items():
        if isinstance(movie_title, list):  # If there are ties
            for movie in movie_title:
                print(f"{genre}\t{movie}\t{avg_rating:.4f}")
        else:
            print(f"{genre}\t{movie_title}\t{avg_rating:.4f}")

if __name__ == "__main__":
    main()
