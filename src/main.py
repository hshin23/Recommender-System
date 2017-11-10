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
                      
    helper.matrixFactorization(testFile);

    

