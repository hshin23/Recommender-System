import numpy as np;

actorsFile = open("../res/additional_files/movie_actors.dat", "r");
directorsFile = open("../res/additional_files/movie_directors.dat", "r");
genresFile = open("../res/additional_files/movie_genres.dat", "r");
movieTagsFile = open("../res/additional_files/movie_tags.dat", "r");
tagsFile = open("../res/additional_files/tags.dat", "r");
testFile = open("../res/additional_files/test.dat", "r");
ttrainFile = open("../res/additional_files/train.dat", "r");
usersFile = open("../res/additional_files/user_taggedmovies.dat", "r");


''' 
* Creating an array for each movie data. 
* Only works for movie data. 
* The index where the element is in the 
  array is the movie id, index=movieid.
* Input is an initialized list, output 
  is the array form of the data.
'''
def createMovieNumpy(file, list):
    for item in file:
        if item.startswith("movieID"):
            continue;
        item = item.replace("\r","").replace("\n","");
        tempList = item.split("\t");
        list[int(tempList[0])-1].append(tempList[1:]);
    
    array = np.array(list);
    return array;

#Actors
actorsList = [[] for i in range(65133)];
actorstArray = createMovieNumpy(actorsFile,actorsList);
 
#Directors
directorsList = [[] for i in range(65133)];
directorsArray = createMovieNumpy(directorsFile,directorsList);
 
#Genres
genresList = [[] for i in range(65133)];
genresArray = createMovieNumpy(genresFile,genresList);

#Movie Tags
movieTagsList = [[] for i in range(65133)];
movieTagsArray = createMovieNumpy(movieTagsFile,movieTagsList);
