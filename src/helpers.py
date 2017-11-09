import numpy as np;
from scipy.sparse import lil_matrix;
from sklearn import preprocessing;
# from django.contrib.sitemaps.views import index;
''' 
* Creating an array for each movie data. 
* Only works for movie data. 
* The index where the element is in the 
  array is the movie id.
* Movies are rows, actors, directors ... 
  are columns. Cells have the actors,
  directors ... info.
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
    

'''
  Making Item to Item Prediction based just on ratings, 
  this will contribute to the main prediction.
  It is really slow right now. 
  This takes the cosine distance between movie columns,
  I will go more in to details on how it works once 
  it is completely working. I am still working on this.
 '''

def matrixFactorization(file):
    from surprise import SVD
    from surprise import Dataset
    from surprise import Reader
    from surprise import evaluate, print_perf
    
    print("Starting File")
    answerFile = open("../res/additional_files/factAnswer.dat", "w");

    file_path = '../res/additional_files/train.dat';
    
    reader = Reader(line_format='user item rating', sep=' ');

    data = Dataset.load_from_file(file_path, reader=reader);
    data.split(n_folds=6);  

    print("Starting Training");
    trainset = data.build_full_trainset();
    

    algo = SVD(n_factors=50, n_epochs=30);
    algo.train(trainset);

    print("Starting Prediction");
    for item in file.readlines()[1:]:
        line = item.replace("\r","").replace("\n","").split(" ");
        user = str(line[0]);
        item = str(line[1]);
        pred = algo.predict(user, item);
        print(np.round(pred.est,2)); 
        answerFile.write(str(np.round(pred.est,1))+"\n");
    perf = evaluate(algo, data, measures=['RMSE', 'MAE']);
    print_perf(perf); 
    answerFile.close();

testFile = open("../res/additional_files/test.dat", "r");
trainFile = open("../res/additional_files/train.dat", "r");                  
matrixFactorization(testFile);

def actorsFact(file):
   from sklearn.metrics import jaccard_similarity_score


    