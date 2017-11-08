'''
* THIS REGARD THIS CODe.
* IT'S WORTHLESS
'''


MOVIE_ACTORS = open("../res/additional_files/movie_actors.dat", 'r')
MOVIE_DIRECTORS = open("../res/additional_files/movie_directors.dat", 'r')
MOVIE_GENRES = open("../res/additional_files/movie_genres.dat", 'r')
MOVIE_TAGS = open("../res/additional_files/movie_genres.dat", 'r')
TAGS = open("../res/additional_files/tags.dat", 'r')
TEST = open("../res/additional_files/test.dat", 'r')
TRAIN = open("../res/additional_files/train.dat", 'r')
USER_TAGGEDMOVIES = open("../res/additional_files/user_taggedmovies.dat", 'r')
UNIQUE_USER_IDS = open("../res/additional_files/unique_user_ids.dat", 'r')
UNIQUE_MOVIE_IDS = open("../res/additional_files/unique_movie_ids.dat", 'r')

OUTPUT_DIR = "../res/res/"


def trimMovieActors():
    NEW_MOVIE_ACTORS = open(OUTPUT_DIR + "movie_actors.dat", 'w')
    NEW_MOVIE_ACTORS.write("movieID actorID actorName ranking\n")

    newid = 1
    for id in UNIQUE_MOVIE_IDS.readlines()[1:]:
        for actor in MOVIE_ACTORS.readlines()[1:]:
            actor = actor.replace('\n', '')
            actorList = actor.split('\t')
            if int(actorList[0]) == int(id):
                NEW_MOVIE_ACTORS.write(str(newid) + "\t" + str(actorList[1] + "\t" + str(actorList[2]) + "\t" + str(actorList[3]) + "\n"))
            else:
                break
        newid += 1

def trimMovieDirectors():
    NEW_MOVIE_DIRECTORS = open(OUTPUT_DIR + "movie_actors.dat", 'w')
    NEW_MOVIE_DIRECTORS.write("movieID actorID actorName ranking\n")
    newid = 1
    for id in UNIQUE_MOVIE_IDS.readlines()[1:]:
        for director in MOVIE_DIRECTORS.readlines()[1:]:
            director = director.replace('\n', '')
            directorList = director.split('\t')
            if int(directorList[0]) == int(id):
                NEW_MOVIE_DIRECTORS.write(str(newid) + "\t" + str(directorList[1] + "\t" + str(directorList[2]) + "\t" + str(directorList[3]) + "\n"))
            else:
                break
        newid += 1

def trimMovieGenres():
    NEW_MOVIE_GENRES = open(OUTPUT_DIR + "movie_genres.dat", 'w')
    NEW_MOVIE_GENRES.write("movieID genre\n")

    newid = 1
    for id in UNIQUE_MOVIE_IDS.readlines()[1:]:
        for genre in MOVIE_GENRES.readlines()[1:]:
            genre = genre.replace("\r", "").replace("\n", "").replace("\t", " ")
            genreList = genre.split()
            if int(genreList[0]) == int(id):
                NEW_MOVIE_GENRES.write(str(newid) + "\t" + str(genreList[1] + "\n"))
        MOVIE_GENRES.seek(0)
        newid += 1
