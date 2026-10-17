#!/usr/bin/env python3

# Student ID:3334661

import sys
import re  # Import regex for title cleaning

# Function to clean movie title
def clean_title(title):
    # Remove special characters (e.g., :, ,, !, etc.)
    title = re.sub(r"[^\w\s]", "", title)  # Keep only alphanumeric and spaces
    # Remove extra spaces
    title = re.sub(r"\s+", " ", title).strip()
    return title

# Read included years.
with open("years.txt", "r") as f:
    included_years = set(f.read().split())  # Use a set for faster lookup

# Read included genres.
with open("genres.txt", "r") as f:
    included_genres = set(f.read().split())  # Use a set for faster lookup

# Check if the lists of years or genres are empty
filter_years = len(included_years) > 0
filter_genres = len(included_genres) > 0

# Process the input line by line.
for line in sys.stdin:
    # Remove trailing spaces and split
    fields = line.strip().split("\t")
    
    if len(fields) < 5:
        continue  # Skip malformed lines

    uid, title, genres, year, rating = [field.strip() for field in fields]  # Strip spaces from all fields

    # Clean movie title
    title = clean_title(title)

    # If year filtering is enabled, check if the movie's year is in the allowed years
    if filter_years and year not in included_years:
        continue  # Skip if the year is not in the allowed years

    try:
        rating = float(rating)  # Convert rating to float
    except ValueError:
        continue  # Skip invalid ratings

    # If genre filtering is enabled, check if the movie's genres are in the allowed genres
    for genre in genres.split("|"):
        if filter_genres and genre not in included_genres:
            continue  # Skip if the genre is not in the allowed genres

        # Output the genre, cleaned movie title, rating, and count (for aggregation)
        print("%s\t%s\t%s\t%s" % (genre, title, rating, 1))
