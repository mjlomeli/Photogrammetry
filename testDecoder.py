import unittest
from decoder import Decoder
import os
from pathlib import Path
import sys
import random
import pickle


FULL_SCAN = False
MEDIUM_SCAN = False


class testDecoder(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.corpus = Corpus()
        self.length = len(self.corpus)
        self.stop_words = set(stopwords.words('english'))
        self.path = Path.cwd() / Path('Test_File.txt')
        self.assertIsNot(None, self.corpus)
        self.assertIsNot(None, self.length)
        self.assertIsNot(None, self.stop_words)
        self.assertIsNot(None, self.path)
        self.bad_files = set([])
        self.good_files = set([])
        self.false_positive_files = None
        with open(self.path, 'w') as file:
            self.assertIsNot(None, file)
            file.write("This is a digital copy copy of a book's that was @ preserved for generations!")

    def test_openFile(self):
        print("Testing:\topenFile()")
        count = 0
        length = self.length if FULL_SCAN else 200 if MEDIUM_SCAN else 10

        self.__printProgressBar(count, length, suffix=str(count) + '/' + str(length) + ' files')
        for i in range(0, length):
            path = self.corpus[i] if FULL_SCAN else self.corpus[random.randint(0, length - 1)]
            self.assertTrue(path.exists())
            self.assertTrue(path.is_file())
            self.tokenizer.openfile(path)
            lentf = len(self.tokenizer.tf.keys())
            lenpos = len(self.tokenizer.pos.keys())
            lentags = len(self.tokenizer.tags.keys())
            self.assertEqual(lentf, lenpos, 'Tf & Pos are different for file:\n' + str(path))
            self.assertEqual(lentf, lentags, 'Tf & Tags are different for file:\n' + str(path))
            self.assertEqual(lentags, lenpos, 'Tags & Pos are different for file:\n' + str(path))
            count += 1
            self.__printProgressBar(count, length, suffix=str(count) + '/' + str(length) + ' files')

    def test_tokenizeWord(self):
        print("Testing:\ttokenize()")
        count = 0
        self.__printProgressBar(count, len(self.stop_words))
        self.assertEqual(self.tokenizer.tokenizeWord('the', 'DT'), None)
        self.assertEqual(self.tokenizer.tokenizeWord('a', 'DT'), None)
        self.assertEqual(self.tokenizer.tokenizeWord('who', 'WP'), None)
        self.assertEqual(self.tokenizer.tokenizeWord('it', 'PRP'), None)
        self.assertEqual(self.tokenizer.tokenizeWord('runners', 'NNS'), 'runner')
        self.assertEqual(self.tokenizer.tokenizeWord('boys', 'NNS'), 'boy')
        self.assertEqual(self.tokenizer.tokenizeWord('titles', 'NNS'), 'title')
        self.assertEqual(self.tokenizer.tokenizeWord('fishes', 'NNS'), 'fish')
        self.assertEqual(self.tokenizer.tokenizeWord('phones', 'NN'), 'phone')
        self.assertEqual(self.tokenizer.tokenizeWord('', 'NNS'), None)
        self.assertEqual(self.tokenizer.tokenizeWord(None, 'NN'), None)
        self.assertEqual(self.tokenizer.tokenizeWord(None, None), None)
        self.assertEqual(self.tokenizer.tokenizeWord('phones', None), 'phone')

        for word in self.stop_words:
            self.assertEqual(self.tokenizer.tokenizeWord(word, 'False'), None)
            count += 1
            self.__printProgressBar(count, len(self.stop_words))

    def test_geTermFreq(self):
        print("Testing:\t__getTermFreq__")
        self.__printProgressBar(0, 1)
        files = [Path.cwd() / Path('Test' + str(i + 1) + '.txt') for i in range(4)]
        with open(files[0], 'w') as f:
            f.write("Should get the term frequency!")
        with open(files[1], 'w') as f:
            f.write("Word's are on the left")
        with open(files[2], 'w') as f:
            f.write("Numbers are on the right")
        with open(files[3], 'w') as f:
            f.write("Finally all in all @ should be working")

        test = [{'get':1, 'term':1, 'frequency':1},
                {'word':1, 'left':1},
                {'number':1, 'right':1},
                {'finally':1, 'work':1}]
        for i in range(len(test)):
            self.tokenizer.openfile(files[i])
            self.assertEqual(test[i], self.tokenizer.getTermFreq())

        for file in files:
            if file.is_file():
                os.remove(file)

        self.__printProgressBar(1, 1)

    def test_getTags(self):
        print("Testing:\tgetTags()")
        self.__printProgressBar(0, 1)
        path = Path.cwd() / Path('Test_Tags.txt')
        with open(path, 'w') as file:
            file.write(r'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"'+\
                       r'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'+\
                       r'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">'+\
                       r'<head>head</head><body>body<p>paragraph</p></body></html>')

        self.tokenizer.openfile(path)
        self.assertEqual(self.tokenizer.tags['head'], ['p'])
        self.assertEqual(self.tokenizer.tags['paragraph'], ['p'])
        if path.is_file():
            os.remove(path)

        self.__printProgressBar(1, 1)

    def test_getParentTags(self):
        print("Testing:\tgetParentTags()")
        self.__printProgressBar(0, 1)
        path = Path.cwd() / Path('Test_Tags.txt')
        elements = None
        with open(path, 'w') as file:
            file.write(r'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"' + \
                       r'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">' + \
                       r'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">' + \
                       r'<head>head</head><body>body<p>paragraph</p></body></html>')
        with open(path, 'rb') as file:
            elements = html.parse(file).getroot().findall('.//p')
        self.tokenizer.openfile(path)

        self.assertEqual(['p'], self.tokenizer.getParentTags(elements[1], 1))
        self.assertEqual(['p'], self.tokenizer.getParentTags(elements[1], 2))
        self.assertEqual(['p'], self.tokenizer.getParentTags(elements[1], 100))
        self.assertEqual([], self.tokenizer.getParentTags(elements[1], 0))
        self.assertEqual([], self.tokenizer.getParentTags(elements[1], -1))
        self.assertEqual([], self.tokenizer.getParentTags(None, 100))
        self.assertEqual([], self.tokenizer.getParentTags(None, None))

        if path.is_file():
            os.remove(path)
        self.__printProgressBar(1, 1)

    def testfalse_positives(self):
        print("Testing:\tFalse Positives")
        with open('false_positives.pickle', 'rb') as f:
            self.false_positive_files = set(pickle.load(f))
            count = 0
            length = len(self.false_positive_files)
            self.__printProgressBar(count, length, str(count) + '/' + str(length) + ' files')
            for file in self.false_positive_files:
                self.tokenizer.openfile(self.corpus[eval(file)])
                for term, value in self.tokenizer.tf.items():
                    try:
                        self.assertLess(len(term), 20, 'Word is too long, check ' + file + 'for word: ' + term + '\n'+term)
                        self.assertLess(value, 4000, 'Word appears too much, check ' + file + ' for word: ' + term + '\n')
                        self.assertLess(len(self.tokenizer.tags[term]), 100, 'Too many tags found, check ' + file + '\n' + term + ', pos=' + str(self.tokenizer.pos[term][:10]) + '...: ' + str(self.tokenizer.tags[term]))
                        self.assertLess(len(self.tokenizer.pos[term]), 700, 'Too many linesources found, check' + file + '\n'+ term + ': ' + str(self.tokenizer.pos[term][:10]) + '...')
                    except Exception as e:
                        print(e)
                        self.bad_files.add(file)
                try:
                    self.assertLess(self.tokenizer.token_count, 10000, "Token count is high, check " + file + '\n' + term)
                except Exception as s:
                    print(s)
                    self.bad_files.add(file)
                count += 1
                self.good_files.add(file)
                self.__printProgressBar(count, length, str(count) + '/' + str(length) + ' files')
            with open('false_positives.pickle', 'wb') as w:
                pickle.dump(self.bad_files, w)

    def __printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        if int(iteration % (total / 100)) == 0 or iteration == total or prefix is not '' or suffix is not '':
            # calculated percentage of completeness
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            # modifies the bar
            bar = fill * filledLength + '-' * (length - filledLength)
            # Creates the bar
            print('\r\t\t{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end='\r')
            # Print New Line on Complete
            if iteration == total:
                print()

    def __del__(self):
        files = [Path.cwd() / Path('Test' + str(i + 1) + '.txt') for i in range(4)]
        for file in files:
            if file.is_file():
                os.remove(file)
        if 'self' in locals() and 'path' in dir(locals()['self']):
            if self.path.exists() and self.path.is_file():
                os.remove(self.path)
        if 'self' in locals() and 'good_files' in dir(locals()['self']) and\
        'self' in locals() and 'bad_files' in dir(locals()['self']) and\
        'self' in locals() and 'false_positive_files' in dir(locals()['self']) and\
                self.false_positive_files is not None:
            self.false_positive_files = self.false_positive_files - self.good_files
            self.false_positive_files = set.union(self.false_positive_files, self.bad_files)

            with open('false_positives.pickle', 'wb') as f:
                pickle.dump(list(self.false_positive_files), f)


class Corpus:

    def __init__(self, directory=None, threshold=3):
        self._search_threshold = threshold
        params = (Path.cwd(), Path('WEBPAGES_RAW'), self._search_threshold)
        if directory is None:
            self.PATH = self.findPathOutward(*params) \
                if self.findPathInward(*params) is None \
                else self.findPathInward(*params)
            if self.PATH is None:
                print("Corpus couldn't find the directory.")
        else:
            self.PATH = Path(directory)
        self.__paths = self.getAllPaths()
        self.__paths.sort()
        self.__file = None
        self.__urls = self.getAllUrls()
        self.__index = 0
        assert self.__len__() > 0, 'The corpus is empty.'

    def findPathInward(self, current, target, limit):
        if limit > 0:
            target_dir = current / target
            if target_dir.exists() and target_dir.is_dir():
                return target_dir
            else:
                for path in current.iterdir():
                    if path.is_dir():
                        found = self.findPathInward(path, target, limit - 1)
                        if found is not None:
                            return found

    def findPathOutward(self, current, target, limit):
        if limit > 0:
            target_dir = current / target
            if target_dir.exists() and target_dir.is_dir():
                return target_dir
            else:
                if current.parent.exists() and current.parent.is_dir():
                    return self.findPathOutward(current.parent, target, limit - 1)

    def getAllPaths(self):
        paths = []
        if self.PATH is not None and self.PATH.is_dir():
            for folder in self.PATH.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        paths.append(file)
        return paths

    def getAllUrls(self):
        book = None
        with open(self.PATH / Path('bookkeeping.json'), 'rb') as f:
            book = json.load(f)
        if book is not None:
            return dict([(val, key) for key, val in book.items()])

    def read_file(self, item):
        path = self.__getitem__(item)
        file = None
        with open(path, 'rb') as f:
            file = f.read()
        return file

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__paths):
            raise StopIteration
        item = self.__paths[self.__index]
        self.__index += 1
        return item

    def __len__(self):
        return len(self.getAllPaths())

    def __getitem__(self, item):
        if isinstance(item, tuple):
            path = self.PATH / Path(str(item[0])) / Path(str(item[1]))
            if path in self.__paths:
                return path
            else:
                raise IndexError(str(item[0]) + "\\" + str(item[1]) + " doesn't exists in corpus.")
        elif isinstance(item, int):
            assert item < len(self.__paths), "The corpus is less than " + str(item) + " length."
            return self.__paths[item]
        elif isinstance(item, str):
            if item in self.__urls:
                return self.PATH / Path(self.__urls[item])
            elif Path(item) in self.__paths:
                return Path(item)
            elif self.PATH / Path(item) in self.__paths:
                return self.PATH / Path(item)
            else:
                print("Error: URL not in bookkeeping.")

    def __contains__(self, item):
        return item in self.__paths

    def __index__(self):
        return self.__paths

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.__paths[key] = Path(value)
        elif isinstance(key, str):
            for i in range(len(self.__paths)):
                if self.__paths[i] == key:
                    self.__paths[i] = Path(value)
        elif isinstance(key, tuple):
            for i in range(len(self.__paths)):
                if str(self.__paths[i]) == str(self.__getitem__(key)):
                    self.__paths[i] = Path(value)

    def __delitem__(self, key):
        if isinstance(key, int):
            self.__paths.remove(self.__paths[key])
        elif isinstance(key, str):
            self.__paths.remove(Path(key))
        elif isinstance(key, tuple):
            self.__paths.remove(self.__getitem__(key))

    def __reversed__(self):
        return reversed(self.__paths)

    def __call__(self, *args, **kwargs):
        if isinstance(args, tuple):
            path = self.PATH / Path(str(args[0])) / Path(str(args[1]))
            if path in self.__paths:
                return path
            else:
                raise IndexError(str(args[0]) + "\\" + str(args[1]) + " doesn't exists in corpus.")
        elif isinstance(args, int):
            assert args < len(self.__paths), "The corpus is less than " + str(args) + " length."
            return self.__paths[args]
        elif isinstance(args, str):
            if args in self.__urls:
                return self.PATH / Path(self.__urls[args])
            elif Path(args) in self.__paths:
                return Path(args)
            elif self.PATH / Path(args) in self.__paths:
                return self.PATH / Path(args)
            else:
                print("Error: URL not in bookkeeping.")

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.__index >= len(self.__paths):
            raise StopAsyncIteration
        item = await self.__paths[self.__index]
        self.__index += 1
        return item


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    if int(iteration % (total / 100)) == 0 or iteration == total or prefix is not '' or suffix is not '':
        # calculated percentage of completeness
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        # modifies the bar
        bar = fill * filledLength + '-' * (length - filledLength)
        # Creates the bar
        print('\r\t\t{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()


def create_false_positives(word_len=20, freq_count=4000, tag_len=100, pos_len=700, token_count=10000):
    """
    Creates files where errors might occur. It creates a list of file paths
    from which the testing will run and give details where errors are likely to occur.
    :param word_len: maximum length a word can be
    :param freq_count: maximum frequency count
    :param pos_len: maximum position values per word
    :param tag_len: maximum number of tags per word
    :param token_count: maximum number of words in the document
    :return:
    """
    count = 0
    t = Tokenizer()
    c = Corpus()
    total = len(c)
    errors = 0
    if total > 0:
        printProgressBar(count, total, 'found: ' + str(errors), str(count) + '/' + str(total) + ' files')
        posts = set([])
        with open('templates/false_positives.pickle', 'w') as f:
            for file in c:
                t.openfile(file)
                if t.token_count > token_count:
                    posts.add(file)
                    errors += 1
                else:
                    length = len(t.tf.keys())
                    words = list(t.tf.keys())
                    if len(words) > 0:
                        i = 0
                        found = False
                        while i < length and not found:
                            if len(words[i]) > word_len:
                                found = True
                                posts.add(file)
                                errors += 1
                            elif t.tf[words[i]] > freq_count:
                                found = True
                                posts.add(file)
                                errors += 1
                            elif len(t.pos[words[i]]) > pos_len:
                                posts.add(file)
                                found = True
                                errors += 1
                            elif len(t.tags[words[i]]) > tag_len:
                                found = True
                                posts.add(file)
                                errors += 1
                            elif words[i] not in t.filters['dictionary']:
                                found = True
                                posts.add(file)
                                errors += 1
                            i += 1
                count += 1
                printProgressBar(count, total, 'found: ' + str(errors), str(count) + '/' + str(total) + ' files')
            posts = list(posts)
            posts.sort()
            pickle.dump(posts, f)

def save_pickle():
    count = 0
    t = Tokenizer()
    c = Corpus()
    total = len(c)
    if total > 0:
        printProgressBar(count, total, str(count) + '/' + str(total) + ' files')
        posts = {}
        with open('templates/tokenizer.pickle', 'w') as f:
            for file in c:
                t.openfile(file)
                posts['tf'] = t.tf
                posts['pos'] = t.pos
                posts['tags'] = t.tags
                posts['size'] = t.size
                posts['token_count'] = t.token_count
                count += 1
                printProgressBar(count, total, str(count) + '/' + str(total) + ' files')
            pickle.dump(posts, f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-f':
            FULL_SCAN = True
        elif sys.argv[1] == '-m':
            MEDIUM_SCAN = True
    print("Testing Tokenizer")
    unittest.main()
