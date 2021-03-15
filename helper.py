import csv
import os
import numpy as np
import tensorflow as tf


SPEECH2VEC_DIR = 'speech2vec/'
S2V_VEC = 's2v-processed.vec'

WORD2VEC_DIR = 'word2vec/vec/'
# W2V_VEC = '100.vec'
W2V_VEC = 'wiki.de.vec'

CROSS_EMBEDDING_DIR = 'cross_embeddings/dumped/debug/best_mapping/'
CE_VEC = 'vectors-des2v.txt'

EMBEDDING_DIMENSION = 100

SYNONYM_DICT = {
    '<laughter>': ['lachen', 'gelächter'],
    '<slightlaughter>': ['lachen', 'gelächter'],
    '<breathing>': ['atmung'],
    '<moaning>': ['stöhnen', 'seufzen', 'gejammer', 'gestöhne'],
    '<contempt>': ['verachtung', 'geringschätzung'],
    '<fumbling>': ['fummelei', 'gefummel', 'linkisch'],
    '<clearingthroat>': ['räuspern'],
    '<coughing>': ['husten'],
    '<singing>': ['singen', 'gesang'],
    '<disgust>': ['ekeln', 'anwidern', 'empören', 'anekeln'],
    '<clicking>': ['klicken']
}


def get_filenames_from_dir(dir):
    trains, vals, tests = [], [], []

    for filename in sorted(os.listdir(dir)):
        fn = filename.split('.')[0]
        if filename.startswith('Train'):
            trains.append(fn)
        elif filename.startswith('Devel'):
            vals.append(fn)
        elif filename.startswith('Test'):
            tests.append(fn)
        else:
            print('Unknown filename: {%s}' % filename)

    return trains, vals, tests


def get_pathlist_from_dir(dir):
    pathlist = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        pathlist.append(path)

    return pathlist


def change_csv_delimiter():
    speech2word_segment_dir = 'speech2vec/segmentation_1'

    pathlist = get_pathlist_from_dir(speech2word_segment_dir)

    for path in pathlist:
        filename = str(path.rsplit('/')[-1])
        fo = open(speech2word_segment_dir + '_new/' + filename, 'w')

        with open(path) as f:
            for row in csv.reader(f):
                new_row = ";".join([x.strip("\";") for x in row])
                fo.write(new_row + '\n')

        fo.close()


def check_tfrecords(dataset_dir='tfrecords/'):
    fn = 'Train_01.tfrecords'

    count = 0
    for example in tf.python_io.tf_record_iterator(dataset_dir + fn):
        count += 1
        
    print('{%d} examples in tfrecord file {%s}' % (count, dataset_dir + fn))


class WordVectorHelper(object):
    def __init__(self, path):
        self.path = path

    def load_vec(self):
        embeddings = []
        embeddings_dict = {}

        id2word = dict()

        with open(self.path, 'r') as f:
            l1 = f.readline().split()
            vocabulary, embedding_size = int(l1[0]), int(l1[1])

            count = 0
            for line in f:
                tmp = line.split()
                try:
                    word, embed = tmp[0], np.array(tmp[1:], dtype=float)

                    if len(embed) == embedding_size:
                        embeddings.append(embed)
                        embeddings_dict[word] = embed
                        id2word[count] = word
                        count += 1
                except:
                    print('Cant process word {%s}' % tmp[0])

        word2id = dict(zip(id2word.values(), id2word.keys()))
        # print('Vocabulary size: %d, embedding size: %d' % (len(embeddings), embedding_size))

        self.embeddings_dict = embeddings_dict
        self.id2word = id2word
        self.word2id = word2id
        self.embeddings = embeddings

        return id2word, word2id, embeddings, embeddings_dict

    def check_for_synonym_in_vec(self):
        # Need to call load_vec() first
        embeddings = self.embeddings_dict

        s_embeddings = dict()
        for k, v in SYNONYM_DICT.items():
            for word in v:
                e = embeddings.get(word, None)
                found = True if e is not None else False
                # print(word, found)

                if found:
                    s_embeddings[k] = e
                    continue

        return s_embeddings

    def get_word_by_embedding(self, vector):
        for i, e in enumerate(self.embeddings):
            if (e == vector).all():
                return self.id2word[i]

        return None


class AVECHelper:
    def __init__(self, space=''):
        if space == 'w2v':
            directory, vec = WORD2VEC_DIR, W2V_VEC
        elif space == 's2v':
            directory, vec = SPEECH2VEC_DIR, S2V_VEC
        else:
            directory, vec = CROSS_EMBEDDING_DIR, CE_VEC

        vec_helper = WordVectorHelper(directory + vec)
        _, _, _, self.embeddings = vec_helper.load_vec()
        self.synonym_embeddings = vec_helper.check_for_synonym_in_vec()

    def process_AVEC(self):
        embeddings = self.embeddings
        synonym_embeddings = self.synonym_embeddings

        directory = 'speech2vec/transcripts/sentence_level_no_filler'
        pathlist = get_pathlist_from_dir(directory)

        vocabulary, unk = set(), dict()
        count, count_unk = 0, 0
        for path in pathlist:
            with open(path, 'r') as f:
                for sentence in f.readlines():
                    punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
                    transcript_clean = sentence.translate(str.maketrans('', '', punctuation))

                    words = transcript_clean.lower().split()
                    for word in words:
                        word = word.strip('\n')
                        vocabulary.add(word)
                        count += 1

                        if synonym_embeddings.get(word, None) is not None:
                            continue

                        if embeddings.get(word, None) is None:
                            unk[word] = unk[word] + 1 if unk.get(word, None) is not None else 1
                            count_unk += 1

            f.close()

        print('---- Process AVEC_DATASET ---')
        print('Unique words: %d' % len(vocabulary))

        unk = sorted(unk.items(), key=lambda x: x[1])
        print('Unique unknown words: %d' % len(unk))
        print(unk)

        print('Total unknown words count: %d, Total words count: %d, Percentage: %.4f'
              % (count_unk, count, (count_unk/count)))


def main():
    s2v_helper = WordVectorHelper(SPEECH2VEC_DIR + S2V_VEC)
    w2v_helper = WordVectorHelper(WORD2VEC_DIR + W2V_VEC)

    s2v_helper.load_vec()
    w2v_helper.load_vec()

    s2v_backup_embeddings = s2v_helper.check_for_synonym_in_vec()
    w2v_backup_embeddings = w2v_helper.check_for_synonym_in_vec()

    check_tfrecords()

    change_csv_delimiter()

