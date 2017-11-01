import numpy as np;
from scipy.sparse import lil_matrix;

actorsFile = open("../res/additional_files/movie_actors.dat", "r");
directorsFile = open("../res/additional_files/movie_directors.dat", "r");
genresFile = open("../res/additional_files/movie_genres.dat", "r");
movieTagsFile = open("../res/additional_files/movie_tags.dat", "r");
tagsFile = open("../res/additional_files/tags.dat", "r");
testFile = open("../res/additional_files/test.dat", "r");
trainFile = open("../res/additional_files/train.dat", "r");
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
        if item.startswith("movieID") or item.startswith("userID"):
            continue;
        item = item.replace("\r","").replace("\n","");
        tempList = item.split("\t");
        list[int(tempList[0])-1].append(tempList[1:]);
    
    array = np.array(list);
    return array;

#Actors
actorsList = [[] for i in range(65134)];
actorstArray = createMovieNumpy(actorsFile,actorsList);
 
#Directors
directorsList = [[] for i in range(65134)];
directorsArray = createMovieNumpy(directorsFile,directorsList);
 
#Genres
genresList = [[] for i in range(65134)];
genresArray = createMovieNumpy(genresFile,genresList);

#Movie Tags
movieTagsList = [[] for i in range(65134)];
movieTagsArray = createMovieNumpy(movieTagsFile,movieTagsList);

#Train Data Matrix
''' 
 * This matrix contains users as rows
   and movies as columns, each cell 
   contains the movie rating.
* It is a lil_sparse matrix because 
  there are a lot of users missing
  and not all user have seen all movies.
'''
def createMatrix(file, matrix):
    for item in file:
        if item.startswith("userID"):
            continue;
        item = item.replace("\r","").replace("\n","");
        tempList = item.split(" ");
        matrix[int(tempList[0]), int(tempList[1])] = tempList[2];
    return matrix;

#Train Data Users info
trainMatrix = lil_matrix((71535, 65134))
trainMatrix = createMatrix(trainFile, trainMatrix);
print(trainMatrix[75,110]);#cell is 4 testing it works.
print(trainMatrix[78,41]);#cell is 4.5 testing it works.


    