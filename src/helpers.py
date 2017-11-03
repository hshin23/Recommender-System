import numpy as np;
from scipy.sparse import lil_matrix;

'''
* Creating an array for each movie data.
* Only works for movie data.
* The index where the element is in the
  array is the movie id, index=movieid.
* Input is an initialized list, output
  is the array form of the data.
'''
def createMovieNumpy(file, list):
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "");
        tempList = item.split("\t");
        list[int(tempList[0]) - 1].append(tempList[1:]);
    array = np.array(list);
    return array;

'''
* This matrix contains users as rows
  and movies as columns, each cell
  contains the movie rating.
* It is a lil_sparse matrix because
  there are a lot of users missing
  and not all user have seen all movies.
'''
def createMatrix(file, matrix):
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        matrix[int(tempList[0]), int(tempList[1])] = tempList[2];
    file.seek(0)
    return matrix;

# returns TF matrix
def tf(file, array, rows, cols):
    tf = lil_matrix((rows, cols))
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        tf[int(tempList[0]), int(tempList[1])] = array[int(tempList[0])][int(tempList[1])] / array[int(tempList[0])].sum()
    file.seek(0)
    return tf

# returns IDF matrix
def idf(file, array, rows, cols):
    tag_counts = [0 for i in range(0, 16530)]
    idf = lil_matrix((rows, cols))
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        tag_counts[int(tempList[1])] += 1
    file.seek(0)
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        idf[int(tempList[0]), int(tempList[1])] = np.log10(65134 / tag_counts[int(tempList[1])])
    file.seek(0)
    return idf

# run tf_idf
def tf_idf(file, array, rows, cols):
    tfMatrix = tf(file, array, rows, cols)
    idfMatrix = idf(file, array, rows, cols)
    tfidfMatrix = lil_matrix((rows, cols))
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        i = int(tempList[0])
        j = int(tempList[1])
        tfidfMatrix[i, j] = tfMatrix[i, j] * idfMatrix[i, j]
    file.seek(0)
    return tfidfMatrix

# returns number of movies rated by user from trainMatrix
def countUserRatings(matrix, userID):
    return matrix.getrowview(userID).getnnz();

# returns sum of movie ratings by user from trainMatrix
def sumUserRatings(matrix, userID):
    return matrix.getrowview(userID).sum(1);
