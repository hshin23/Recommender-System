# Recommender System

CS484 HW4 for Fall 2017.

## Introduction

Recommender system uses the rating information and side information to predict correct recommendations of movies to its users.

## Getting Started

### Dependencies

A python implementation where `python --version >= 3`
 - numpy
 - nltk
 - enchant

### Running

This is the hypothetical way to run the scripts. (not implemented yet)
  ~~~ sh
  $ git clone https;//github.com/hshin23/Recommender-System
  $ cd Recommender-System
  $ ./run.sh
  ~~~

### To Contribute

  ~~~ sh
  $ git clone https://github.com/hshin23/Recommender-System
  $ cd Recommender-System
  $ git checkout <EXISTING-BRANCH-TO-WORK-ON>
  or
  $ git checkout -b <NEW-BRANCH-FOR-NEW-FEATURES>

## File Descriptions
* train.dat
This file containts the rating of a user for a give movie.

*test.dat
This file contains user movie pairs but no rating (Your goal is to predict these ratings for user-movie pairs)

* movie_genres.dat
This file contains the genres of the movies.

* movie_directors.dat
This file contains the directors of the movies.

* movie_actors.dat
This file contains the main actores and actresses of the movies. A ranking is given to the actors of each movie according to the order in which  they appear on the movie IMDb cast web page.

* tags.dat
This file contains the set of tags available in the dataset.

* user_taggedmovies.dat 
These files contain the tag assignments of the movies provided by each particular user.

* movie_tags.dat
This file contains the tags assigned to the movies, and the number of times  the tags were assigned to each movie.

test.dat: Test set consistting of user-movie pairs for which you need to produce the ratings

example_entry.dat: A sample submission with 71299 entries in the range of 1-5
