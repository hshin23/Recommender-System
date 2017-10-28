'''
container: container for all simple objects in the given data
'''

__all__ = ["TestSet", "TagSet", "TrainSet"]


# TestSet contains multiple Tests
# TODO: come up with better names
class TestSet:
    def __init__(self, path):
        self.tests = self.__load_tests(path)

    def __load_tests(self, path):
        tests = []
        with open(path, 'r') as stream:
            for line in stream.readlines()[1:]:
                tmp = line.split()
                tests.append(self.TestObject(tmp[0], tmp[1]))
        return tests

    def __str__(self):
        literal = ""
        for test in self.tests:
            literal += str(test) + "\n"
        return literal

    def __repr__(self):
        literal = ""
        for test in self.tests:
            literal += str(test) + "\n"
        return literal

    class TestObject:
        def __init__(self, user_id, movie_id):
            self.user_id = int(user_id)
            self.movie_id = int(movie_id)

        def __str__(self):
            return "[" + str(self.user_id) + ", " + str(self.movie_id) + "]"

        def __repr__(self):
            return "[" + str(self.user_id) + ", " + str(self.movie_id) + "]"


class TagSet:
    def __init__(self, path):
        self.tags = self.__load_tags(path)

    def __load_tags(self, path):
        tags = []
        with open(path, 'r') as stream:
            for line in stream.readlines()[1:]:
                tmp = line.split()
                tags.append(self.TagObject(tmp[0], tmp[1:]))
        return tags

    def __str__(self):
        literal = ""
        for tag in self.tags:
            literal += str(tag) + "\n"
        return literal

    def __repr__(self):
        literal = ""
        for tag in self.tags:
            literal += str(tag) + "\n"
        return literal

    class TagObject:
        def __init__(self, tag_id, values):
            self.tag_id = tag_id
            self.values = values

        def __str__(self):
            return "[" + str(self.tag_id) + ", " + str(self.values) + "]"

        def __repr__(self):
            return "[" + str(self.tag_id) + ", " + str(self.values) + "]"


class TrainSet:
    def __init__(self, path):
        self.trains = self.__load_trains(path)

    def __load_trains(self, path):
        data = []
        with open(path, 'r') as stream:
            for line in stream.readlines()[1:]:
                tmp = line.split()
                data.append(self.Tag(tmp[0], tmp[1], tmp[2]))
        return data

    def __str__(self):
        literal = ""
        for train in self.trains:
            literal += str(train) + "\n"
        return literal

    def __repr__(self):
        literal = ""
        for train in self.trains:
            literal += str(train) + "\n"
        return literal

    class TrainObject:
        def __init__(self, user_id, movie_id, rating):
            self.id = int(user_id)
            self.movie_id = int(movie_id)
            self.rating = float(rating)

        def __str__(self):
            return "[" + str(self.user_id) + ", " + str(self.movie_id) + ", " + str(self.rating) + "]"

        def __repr__(self):
            return "[" + str(self.user_id) + ", " + str(self.movie_id) + ", " + str(self.rating) + "]"
