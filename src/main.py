import helpers as helper
from scipy.sparse import lil_matrix;

COUNT_ACTORS = 65134
COUNT_MOVIES = 65134
COUNT_TAGS   = 16530
COUNT_TRAINS = 71535

if __name__ == "__main__":
    actorsFile = open("../res/additional_files/movie_actors.dat", "r");
    directorsFile = open("../res/additional_files/movie_directors.dat", "r");
    genresFile = open("../res/additional_files/movie_genres.dat", "r");
    movieTagsFile = open("../res/additional_files/movie_tags.dat", "r");

    tagsFile = open("../res/additional_files/tags.dat", "r");
    testFile = open("../res/additional_files/test.dat", "r");
    trainFile = open("../res/additional_files/train.dat", "r");
    usersFile = open("../res/additional_files/user_taggedmovies.dat", "r");

    # Actors
    actorsList = [[] for i in range(0, 65134)]
    actorstArray = helper.createMovieNumpy(actorsFile, actorsList);

    # Directors
    directorsList = [[] for i in range(0, 65134)]
    directorsArray = helper.createMovieNumpy(directorsFile, directorsList);

    # Genres
    genresList = [[] for i in range(0, 65134)]
    genresArray = helper.createMovieNumpy(genresFile, genresList);

    # Movie Tags
    movieTagsList = [[] for i in range(0, 65134)]
    movieTagsArray = helper.createMovieNumpy(movieTagsFile, movieTagsList);

    # Reset file pointer
    actorsFile.seek(0)
    directorsFile.seek(0)
    genresFile.seek(0)
    movieTagsFile.seek(0)

#     # Train Data Users info
#     trainMatrix = lil_matrix((71535, 65134))
#     trainMatrix = helper.createMatrix(trainFile, trainMatrix);
# 
#     # Tags Matrix
#     tagsMatrix = lil_matrix((65134, 16530))
#     tagsMatrix = helper.createMatrix(movieTagsFile, tagsMatrix);
#     tagsMatrix = helper.tf_idf(movieTagsFile, tagsMatrix.toarray(), COUNT_MOVIES, COUNT_TAGS)
#  
#     #     Find similar movies according to tag
#     distMatrix = helper.findSimilarityByTags(movieTagsFile, tagsMatrix)
#     print(distMatrix[1, 1])
#     print(distMatrix[1, 2])
                      
    matrixFactorization(testFile);
    

