#!/usr/bin/env python3
# combiner.py
# Student ID: 3334661
import sys
import logging
from collections import defaultdict

# Set up logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

def main():
    # Dictionary to store aggregated ratings and counts for each (genre, movie_title) pair
    genre_movie_ratings = defaultdict(lambda: [0.0, 0])  # { (genre, movie_title) : [sum_ratings, count] }
    
    for line in sys.stdin:
        fields = line.strip().split('\t')
        if len(fields) < 4:
            logging.error(f"Malformed line skipped: {line.strip()}")
            continue  # Skip malformed lines
        
        genre, movie_title, rating, count = fields
        
        try:
            rating = float(rating)  # Convert rating to float
            count = int(count)       # Convert count to int
        except ValueError:
            logging.error(f"Invalid data skipped: {line.strip()}")
            continue  # Skip lines with invalid data
        
        # Aggregate ratings and counts for this (genre, movie_title) pair
        genre_movie_ratings[(genre, movie_title)][0] += rating  # Sum up ratings
        genre_movie_ratings[(genre, movie_title)][1] += count   # Sum up counts
    
    # Emit the fully aggregated results for each (genre, movie_title) pair
    for (genre, movie_title), (total_rating, total_count) in genre_movie_ratings.items():
        print(f"{genre}\t{movie_title}\t{total_rating}\t{total_count}")

if __name__ == "__main__":
    main()
