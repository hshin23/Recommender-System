import numpy as np;
from scipy.sparse import lil_matrix, coo_matrix;
import progressbar

'''
TODO
1. TF_IDF of tags
2. Elimination of tags by tf_idf rank
'''

'''
* returns list of unique id
* file: file to read in from
* col:  index of the column that contains ids. ranges from 0 ... N
'''
def get_unique_ids(file, col):
    id_list = []
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        current_id = int(tempList[col])
        try:
            id_list.index(current_id)
        except:
            id_list.append(current_id)
    file.seek(0)
    return id_list


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
def createMatrix(file, row, col):
    matrix = lil_matrix((row, col))
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        if len(tempList) == 2:
            matrix[int(tempList[0]), int(tempList[1])] = 1;
        if len(tempList) == 3:
            matrix[int(tempList[0]), int(tempList[1])] = tempList[2];
    file.seek(0)
    return matrix;

def createCooMatrix(file):
    row = []
    col = []
    data = []
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        if len(tempList) == 3:
            row.append(int(tempList[0]))
            col.append(int(tempList[1]))
            data.append(float(tempList[2]))
        else:
            row.append(int(tempList[0]))
            col.append(int(tempList[1]))
            data.append(0)
    file.seek(0)
    return coo_matrix((data, (row, col)))

'''
* Term-Frequency
* term_count / total_words_in_a_doc
* returns TF matrix
'''
def tf(file, array, rows, cols):
    tf = lil_matrix((rows, cols))
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        tf[int(tempList[0]), int(tempList[1])] = array[int(tempList[0])][int(tempList[1])] / array[int(tempList[0])].sum()
    file.seek(0)
    return tf

'''
* Inverse-Document Frequency
* log10(numDocuments / numTermAppearences)
* Note: currently uses 65134 as number of
         documents. it may be better to use
        number of *actual* documents
* TODO: research/ask around about what number
        to use
* returns IDF matrix
'''
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

'''
* Term-Frequency Inverse-Doc Freq
* returns lil_matrix containiing
  new tag_weight
'''
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

'''
* TODO: Use tag_weights after running tf_id/
        to somehow make the distance better.
        And also save that into a file.
* computes similarity
* wrapper for jaccard
'''
def findSimilarityByTags(file, out, matrix):
    distMatrix = lil_matrix((65134, 65134))
    id_list = []
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        if len(id_list) == 0:
            id_list.append(int(tempList[0]))
        else:
            if id_list[len(id_list) - 1] != int(tempList[0]):
                id_list.append(int(tempList[0]))
    file.seek(0)
    for id in id_list:
        distMatrix = jaccard(file, out, matrix, distMatrix, id, id_list)
        # print(str(id) + "/65130")
    return distMatrix

# computes jaccard distance between id & other_ids
# TODO: make union/inter calculation faster..
def jaccard(file, out, matrix, distMatrix, id, id_list):
    union = count_unique(file, matrix, id, id_list)             # count unique bettwen id & other_ids
    inter = count_duplicate(file, matrix, id, id_list)          # count duplicates between id & other_ids
    for item in id_list:
        # skip if computed already
        if distMatrix[id, item] != 0:
            continue
        else:
            # compute jaccard distance
            jaccard_index = inter[item] / union[item]
            jaccard_dist = 1 - jaccard_index
            distMatrix[id, item] = jaccard_dist
            # reset file pointer
            file.seek(0)
            # write to outfile
            if jaccard_dist != 1:
                out.write(str(id) + "\t" + str(item) + "\t" + str(jaccard_dist) + "\n")
    return distMatrix

def count_unique(file, matrix, id, id_list):
    sum_id = 0
    sum_list = [0 for i in range(0, 65134)]
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        if matrix[id, int(tempList[1])] == 0:
            sum_list[int(tempList[0])] += 1
        if int(tempList[0]) == id:
            sum_id += 1
    file.seek(0)
    sum_list = [x + sum_id for x in sum_list]
    return sum_list

def count_duplicate(file, matrix, id, id_list):
    inter = [0 for i in range(0, 65134)]
    for item in file.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        if matrix[id, int(tempList[1])] > 0:
            inter[int(tempList[0])] += 1
    file.seek(0)
    return inter

# returns number of movies rated by user from trainMatrix
def countUserRatings(matrix, userID):
    return matrix.getrowview(userID).getnnz();

# returns sum of movie ratings by user from trainMatrix
def sumUserRatings(matrix, userID):
    return matrix.getrowview(userID).sum(1);

# cut down by top K
def cutdown(matrix, orig, out, K):
    matrix = matrix.toarray()
    last_id = 0
    for item in orig.readlines()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        i = int(tempList[0])
        if i == last_id:
            continue
        for j in range(0, K):
            if matrix[i].sum() != 0:
                _index = matrix[i].tolist().index(matrix[i].max())
                matrix[i][_index] = 0
                out.write(str(i) + "\t" + str(_index) + "\n")
        last_id = i
    orig.seek(0)
