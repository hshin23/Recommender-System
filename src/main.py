import numpy as np;

actorsFile = open("../res/additional_files/movie_actors.dat");
directorsFile = open("../res/additional_files/movie_directors.dat");
genresFile = open("../res/additional_files/movie_genres.dat");
movieTagsFile = open("../res/additional_files/movie_tags.dat");
tagsFile = open("../res/additional_files/tags.dat");
testFile = open("../res/additional_files/test.dat");
ttrainFile = open("../res/additional_files/train.dat");
usersFile = open("../res/additional_files/user_taggedmovies.dat");




''' 
Creating the array of whatever the input is. 
Only works for movie data. The index is the movie id
Input is an initialized list, output is the array.
'''
def createMovieNumpy(file, list):
    for item in file:
        if item.startswith("movieID"):
            continue;
        item = item.replace("\r","").replace("\n","");
        tempList = item.split("\t");
        list[int(tempList[0])-1].append(tempList[1:]);
    
    array = np.array(list);
#     print(array);
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

#Tags
movieTagsList = [[] for i in range(65133)];
movieTagsArray = createMovieNumpy(movieTagsFile,movieTagsList);
