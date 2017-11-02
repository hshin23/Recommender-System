import numpy as np;
from scipy.sparse import lil_matrix;
from sklearn.metrics.pairwise import cosine_similarity;
from statistics import mean;

actorsFile = open("../res/additional_files/movie_actors.dat", "r");
directorsFile = open("../res/additional_files/movie_directors.dat", "r");
genresFile = open("../res/additional_files/movie_genres.dat", "r");
movieTagsFile = open("../res/additional_files/movie_tags.dat", "r");
tagsFile = open("../res/additional_files/tags.dat", "r");
testFile = open("../res/additional_files/test.dat", "r");
trainFile = open("../res/additional_files/train.dat", "r");
usersFile = open("../res/additional_files/user_taggedmovies.dat", "r");
# 
# ''' 
# * Creating an array for each movie data. 
# * Only works for movie data. 
# * The index where the element is in the 
#   array is the movie id.
# * Movies are rows, actors, directors ... 
#   are columns. Cells have the actors,
#   directors ... info.
# * Input is an initialized list, output 
#   is the array form of the data.
# '''
# def createMovieNumpy(file, list):
#     for item in file:
#         if item.startswith("movieID"):
#             continue;
#         item = item.replace("\r","").replace("\n","");
#         tempList = item.split("\t");
#         list[int(tempList[0])].append(tempList[1:]);
#     array = np.array(list);
#     return array;
# 
'''
 TODO:  do jaccard distance on each of this with the test user movie info to
        find a prediction. According to how well it does on predicting
        we give it a weight of to contribute to the main answer.
        We can move these and the ^ top ones to a different file.
'''
# #Actors
# actorsList = [[] for i in range(65134)];
# actorstArray = createMovieNumpy(actorsFile,actorsList);
#  
# #Directors
# directorsList = [[] for i in range(65134)];
# directorsArray = createMovieNumpy(directorsFile,directorsList);
#  
# #Genres
# genresList = [[] for i in range(65134)];
# genresArray = createMovieNumpy(genresFile,genresList);
# 
# #Movie Tags
# movieTagsList = [[] for i in range(65134)];
# movieTagsArray = createMovieNumpy(movieTagsFile,movieTagsList);


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
  
#Train Data Matrix
trainMatrix = lil_matrix((71535, 65134));
trainMatrix = createMatrix(trainFile, trainMatrix);
print(trainMatrix[75,110]);#cell is 4 testing it works.
print(trainMatrix[78,41]);#cell is 4.5 testing it works.


'''
   Rows are users, the columns are movies, cells the moveiId. 
'''
def createTestList(file, list):
    for item in file:
        if item.startswith("userID"):
            continue;
        item = item.replace("\r","").replace("\n","");
        tempList = item.split(" ");
        list[int(tempList[0])].append(int(tempList[1]));
    return list;
testList = [[] for i in range(71535)];
testList = createTestList(testFile, testList);


'''
  Making Item to Item Prediction based just on ratings, 
  this will contribute to the main prediction.
  It is really slow right now. 
  This takes the cosine distance between movie columns,
  I will go more in to details on how it works once 
  it is completely working. I am still working on this.
'''
cosineDistanceList = [];
fiveMostSimilar = [];
movieLocation = []
matrixCloumns = trainMatrix.T
answerFile = open("answerFile.dat", "w");
for testMovies in testList:#The test list row contains more than one movie to rate for each user.
    for testMovie in testMovies:
        if testMovies:#Start with just one movie to predict.
            for movie in matrixCloumns:#Movie from matrix of train data.
                if movie.nnz != 0:#Only if not empty.
                     movieLocation.append(movie);#Saving the column of the movie to know the position.
                     cosineDistanceList.append(cosine_similarity( movie, matrixCloumns[testMovie])[0]);#Saving the distance from that column and the test data movie column.
            for ind in range(5):#To select only the 5 most similar to the test data movie column.
                 temp = movieLocation[cosineDistanceList.index(min(cosineDistanceList))];#Testing.
                 print(str(temp))
                 fiveMostSimilar.append(movieLocation[cosineDistanceList.index(min(cosineDistanceList))].toarray());
                 cosineDistanceList.remove(min(cosineDistanceList));
            if fiveMostSimilar:
                print(fiveMostSimilar);
                print(np.mean(fiveMostSimilar));#The answer is the mean of the most similar movie columns.
                answerFile.write(str(np.mean(fiveMostSimilar))+"\n");
        cosineDistanceList = [];
        fiveMostSimilar = []
    
    
    
    
    
    
    
    
    