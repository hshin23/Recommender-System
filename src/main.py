import helpers as helper
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.model_selection import train_test_split
import progressbar
import time
import numpy as np
import math

COUNT_ACTORS = 65134
COUNT_MOVIES = 65134
COUNT_TAGS   = 16530
COUNT_USERS = 71535

COUNT_USERS_UNIQUE = 2114
COUNT_MOVIES_UNIQUE = 10001

__FILE_TEST = open("../res/additional_files/test.dat", "r");
__FILE_TRAIN = open("../res/additional_files/train.dat", "r");
FILE_MOVIExTAG = open("../res/additional_files/movie_tags.dat", "r")
FILE_MOVIExMOVIE = open("../res/additional_files/distbytag.dat", "r")

MOVIE_IDS = []
USER_IDS = []

def generate_default_IDs():
    print("Generating unique movie ids from train file...")
    MOVIE_IDS_TRAIN = sorted(helper.get_unique_ids(__FILE_TRAIN, 1))
    print("Generating unique movie ids from test file...")
    MOVIE_IDS_TEST = sorted(helper.get_unique_ids(__FILE_TEST, 1))
    print("Generating unique user ids from train file...")
    USER_IDS_TRAIN = sorted(helper.get_unique_ids(__FILE_TRAIN, 0))
    print("Generating unique user ids from test file...")
    USER_IDS_TEST = sorted(helper.get_unique_ids(__FILE_TEST, 0))
    print("Writing to disk...")
    with open("../res/additional_files/my_files/unique_movie_ids.dat", 'w') as out:
        out.write("movie_id\n")
        _ = sorted(list(set(MOVIE_IDS_TRAIN + MOVIE_IDS_TEST)))
        for elem in _:
            out.write(str(elem) + "\n")
    with open("../res/additional_files/my_files/unique_train_movie_ids.dat", 'w') as out:
        out.write("movie_id\n")
        for elem in MOVIE_IDS_TRAIN:
            out.write(str(elem) + "\n")
    with open("../res/additional_files/my_files/unique_test_movie_ids.dat", 'w') as out:
        out.write("movie_id\n")
        for elem in MOVIE_IDS_TEST:
            out.write(str(elem) + "\n")
    with open("../res/additional_files/my_files/unique_user_ids.dat", 'w') as out:
        out.write("user_id\n")
        for elem in USER_IDS_TRAIN:
            out.write(str(elem) + "\n")
    import_default_IDs()

def import_default_IDs():
    with open("../res/additional_files/my_files/unique_user_ids.dat", 'r') as file:
        for line in file.readlines()[1:]:
            line = line.split('\n')
            USER_IDS.append(int(line[0]))
    with open("../res/additional_files/my_files/unique_movie_ids.dat", 'r') as file:
        for line in file.readlines()[1:]:
            line = line.split('\n')
            MOVIE_IDS.append(int(line[0]))

def export_dense(inPath, outPath):
    with open(inPath, 'r') as inFile, open(outPath, 'w') as outFile:
        outFile.write("user_id movie_id rating\n")
        for line in inFile.readlines()[1:]:
            line = line.replace("\r", "").replace("\n", "").replace("\t", " ");
            tempList = line.split(" ");
            if len(tempList) == 2:
                outFile.write(str(USER_IDS.index(int(tempList[0])) + 1) + "\t" + tempList[1] + "\n")
            if len(tempList) == 3:
                outFile.write(str(USER_IDS.index(int(tempList[0])) + 1) + "\t" + tempList[1] + "\t" + tempList[2] + "\n")

def compute_pearson(userA, userB, train):
    movies_in_both = get_both(userA, userB, train)
    if len(movies_in_both) == 0:
        return [userB, -1]
    else:
        ra = get_average(userA, movies_in_both, train)
        rb = get_average(userB, movies_in_both, train)
        top = 0
        bota = 0
        botb = 0
        for i in range(0, len(movies_in_both)):
            rap = train[userA][movies_in_both[i]]
            rbp = train[userB][movies_in_both[i]]
            top += ((rap - ra) * (rbp - rb))
            bota += ((rap - ra) * (rap - ra))
            botb += ((rbp - rb) * (rbp - rb))
        bota = np.sqrt(bota)
        botb = np.sqrt(botb)
        if bota == 0 or botb == 0:
            # print("userA = " + str(userA) + "\tuserB = " + str(userB) + "\ttop = " + str(top) + "\tbota = " + str(bota) + "\tbotb = " + str(botb))
            # print("movies_in_both = " + str(movies_in_both))
            return 0
        sim = top / (bota * botb)
        # print("top = " + str(top) + "\tbota = " + str(bota) + "\tbotb = " + str(botb))
        return [userB, sim]

# returns list of movie_ids rated by both users
def get_both(userA, userB, train):
    a = []
    b = []
    both = []
    for i, rating in enumerate(train[userA]):
        if rating != 0:
            a.append(i)
    for i, rating in enumerate(train[userB]):
        if rating != 0:
            b.append(i)
            try:
                a.index(i)
                both.append(i)
            except:
                continue
    return both

def get_average(userA, movies, train):
    # what happens when there are no similar movies
    temp = []
    for movie in movies:
        temp.append(train[userA][movie])
    return np.mean(temp)

def predict(userA, movie, sims, train):
    temp = []
    for elem in train[userA]:
        if elem != 0:
            temp.append(elem)
    ra = np.mean(temp)
    top = 0
    bot = 0
    for sim in sims:
        # print(sim[1])
        userB = sim[0]
        movies_in_both = get_both(userA, userB, train)
        if len(movies_in_both) == 0:
            continue
        rbp = train[userB][movie]
        rb = get_average(userB, movies_in_both, train)  ## problemo -> NaN
        top += (rbp - rb) * sim[1]
        bot += sim[1]
    ans = ra + (top / bot)
    print("ans = " + str(ans))
    return ans

def my_round(num):
    return round(num * 2) / 2


if __name__ == "__main__":
    if input("Generate default ID lists? (+40sec) [y/All]  ") == 'y':
        generate_default_IDs()
    else:
        import_default_IDs()

    # densen the user_ids from 71535 -> 2113
    if input("Export dense train and test data? [y/All]  ") == 'y':
        export_dense("../res/additional_files/train.dat", "../res/additional_files/my_files/densed_train.dat")
        export_dense("../res/additional_files/test.dat", "../res/additional_files/my_files/densed_test.dat")

    # declare/initialize variables
    trainMatrix = helper.createMatrix(open("../res/additional_files/my_files/densed_train.dat", 'r'), 2114, 65134)
    trainArray = trainMatrix.toarray()
    testFile = open("../res/additional_files/my_files/densed_test.dat", 'r')
    outputFile = open("../res/additional_files/my_files/output.dat", 'w')
    simMatrix = lil_matrix((2114, 2114))

    # create average list
    avg_list = [[] for i in range(0, 2114)]
    for i in range(1, 2114):
        sigma = trainMatrix.getrow(i)[0]
        sigma = sigma[sigma != 0]
        avg_list[i] = sigma.sum() / sigma.size

    count = 0
    for test in testFile.readlines()[1:]:
        time.sleep(0.05)
        test = test.replace("\r", "").replace("\n", "").replace("\t", " ")
        test = test.split(" ")
        userA = int(test[0])
        movieID = int(test[1])
        rA = avg_list[userA]
        col_array = trainMatrix.getcol(movieID)
        # for mean -- this is dumb, erase later
        col_array = col_array[col_array != 0]
        mean = my_round(col_array.sum() / col_array.size)
        outputFile.write(str(mean) + '\n')
        print(count)
        count += 1
        '''
        for userB, elem in enumerate(col_array):
            sum_top = 0
            sum_bota = 0
            sum_botb = 0
            if elem != 0:
                sigma = trainMatrix.getrow(userB)[0]
                sigma = sigma[sigma != 0]
                rB = (sigma.sum() - trainMatrix[userB, movieID]) / (sigma.size - 1)
                intersection = get_both(userA, userB, trainArray)
                for item in intersection:
                    rAP = trainMatrix[userA, item]
                    rBP = trainMatrix[userB, item]
                    sum_top += (rAP - rA) * (rBP - rB)
                    sum_bota += (rAP - rA) * (rAP - rA)
                    sum_botb += (rBP - rB) * (rBP - rB)
                sum_bota = np.sqrt(sum_bota)
                sum_botb = np.sqrt(sum_botb)
                sum_bot = sum_bota * sum_botb
                sim = sum_top / sum_bot
                if simMatrix[userA, userB] != 0:
                    print(simMatrix[userA, userB])
                    print(sim)
                simMatrix[userA, userB] = sim
                print("similarity[" + str(userA) + ", " + str(userB) + "] = " + str(simMatrix[userA, userB]))
        '''





    '''
    # declare/initialize variables
    __trainMatrix = helper.createMatrix(open("../res/additional_files/my_files/densed_train.dat", 'r'), 2114, 65)
    trainArray = __trainMatrix.toarray()
    testFile = open("../res/additional_files/my_files/densed_test.dat", 'r')
    outputFile = open("../res/additional_files/my_files/output.dat", 'w')

    for item in testFile.readlins()[1:]:
        item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
        tempList = item.split(" ");
        userA = int(tempList[0])
        movie = int(tempList[1])
    output = open("../res/additional_files/my_files/output.txt", 'w')

    with open("../res/additional_files/my_files/densed_test.dat", 'r') as testFile:
        for item in testFile.readlines()[1:]:
            item = item.replace("\r", "").replace("\n", "").replace("\t", " ");
            tempList = item.split(" ");
            userA = int(tempList[0])
            movie = int(tempList[1])

            # find users who rated movie above
            sims = []
            for user in range(1, 2114):
                if trainArray[user][movie] != 0:
                    userB = user
                    # do pearson
                    val = compute_pearson(userA, userB, trainArray)
                    if val != 0:
                        sims.append(val)
            # run prediction on simMatrix[userA]
            # sims contains [userB, similarity(userA, userB)]
            val = predict(userA, movie, sims, trainArray)
            # output.write(str(val) + "\n")

    '''
