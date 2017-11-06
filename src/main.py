import helpers as helper
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.model_selection import train_test_split

COUNT_ACTORS = 65134
COUNT_MOVIES = 65134
COUNT_TAGS   = 16530
COUNT_TRAINS = 71535

FILE_TEST = open("../res/additional_files/test.dat", "r");
FILE_TRAIN = open("../res/additional_files/train.dat", "r");
FILE_MOVIExTAGS = open("../res/additional_files/movie_tags.dat", "r")

# MOVIE_IDS_TRAIN = helper.get_unique_ids(FILE_TRAIN, 1)
# MOVIE_IDS_TEST = helper.get_unique_ids(FILE_TEST, 1)

if __name__ == "__main__":
    movie = lil_matrix((71536, 65135))
    movie = helper.createMatrix(FILE_TRAIN, movie)
