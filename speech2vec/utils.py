"""
Train a speech embedding spaces by using Speech2Vec (German-corpus)
"""
import csv
import os
import sys
import random
sys.path.append(".")

from word2vec.eval import evaluate
from helper import get_pathlist_from_dir

AVEC_DIR = '/Users/anhnguyen/Downloads/projects/imperial-project/AVEC2017_SEWA'
TRANSCRIPT_DIR = AVEC_DIR + '/transcriptions/'


def clean_pretrained_vec():
    _, embeddings, id2word, _ = load_vec()
    dim = len(embeddings[0])
    write_f(id2word, dim, embeddings)


def load_vec():
    # Open raw vec file
    dirname = os.path.dirname(__file__)
    filename = dirname + '/s2v.vec'
    fi = open(filename, 'r')

    # Get vocab_size, embedding dimension
    vec = fi.readline().split()
    vocab_size, dim = int(vec[0]), int(vec[1])

    # Process raw vec
    # (1) Lowercase, (2) Remove non-alpha words
    dictionary, embeddings = set(), []
    id2word = {}

    id, count_overlapping, count_nonalpha = 0, 0, 0
    nonalphas, overlap_dict = list(), dict()

    for line in fi:
        word, embed = line.split(' ', 1)
        processed_word = word.lower()

        # Remove non-alpha words
        processed_word = ''.join(e for e in processed_word if e.isalpha())
        if processed_word == '':
            count_nonalpha += 1
            nonalphas.append(word)
            continue

        # Add unique words to dictionary
        if processed_word in dictionary:
            count_overlapping += 1
            overlap_dict[processed_word].append(word)
            # print('Error: Found word {%s} in dictionary while processing word {%s}' % (processed_word, word))
        else:
            dictionary.add(processed_word)
            overlap_dict[processed_word] = [word]

            id2word[id] = processed_word
            embeddings.append([float(x) for x in embed.split()])
            id += 1

    # Get a list of repeated words after cleaned
    for processed_word in overlap_dict.copy():
        similar_words = overlap_dict[processed_word]
        if len(similar_words) == 1:
            del overlap_dict[processed_word]

    print('Vocabulary size: %d' % vocab_size)
    print('Vocabulary size after processed: %d' % (len(dictionary)))
    print('Non-alpha words: %d' % count_nonalpha)
    print('Overlapping words: %d' % count_overlapping)
    print('Unique overlapping words: %d' % len(overlap_dict))

    count = [(w, len(l)) for w, l in overlap_dict.items()]
    count = sorted(count, key=lambda x: x[1])
    print('Most common repeated words: ')
    print(count[-10:])
    # count_tmp = [(x[0], x[1]) for x in count if x[1] > 3]  # 3578 words = 2, 1933 words = 3, 889 others

    # _, eval_words = random.choice(list(overlap_dict.items()))
    # evaluate(words=eval_words, directory=dirname, file='/s2v.vec')

    word2id = dict(zip(id2word.values(), id2word.keys()))
    # Write refined embeddings to file
    return dictionary, embeddings, id2word, word2id


def write_f(id2word, dim, embeddings):
    dirname = os.path.dirname(__file__)
    file_output = dirname + '/s2v-processed.vec'

    fo = open(file_output, 'w')
    fo.write(str(len(id2word)) + ' ' + str(dim) + '\n')

    for i in range(len(id2word)):
        w = id2word[i]
        e = ' '.join(str(x) for x in embeddings[i])
        fo.write(w + ' ' + e + '\n')

    fo.close()


# Force alignment speech-to-text (based on transcripts) (Option_1)
# (Option 2) in speech2word_mapping.py
def force_alignment_STT():
    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/mappings/'

    pathlist = get_pathlist_from_dir(TRANSCRIPT_DIR)

    for path in pathlist:
        # Read transcript file
        fi = open(path, newline='')
        reader = csv.reader(fi, delimiter=';', quotechar='|')

        # Get filename
        p = path.rsplit('/', 1)
        filename = str(p[-1])[:-4]

        # Open file out
        fo = open(output_dir + filename + '.csv', 'w')

        sent_num, word_num = 0, 0
        for row in reader:
            sent_num += 1

            transcript = row[2].strip('\'')
            if transcript == '<filler>':
                continue

            # Clean sentence
            punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
            transcript_clean = transcript.translate(str.maketrans('', '', punctuation))

            words = transcript_clean.lower().split()

            if len(words) == 0:
                print(filename, sent_num)
                continue

            # Force alignment to map word and time
            start, end = float(row[0]), float(row[1])
            time_gap = (end - start) / len(words)
            st, et = round(start, 6), round(start + time_gap, 6)

            for word in words:
                if word == '<filler>':
                    print('{%s} contains "<filler>" at sentence %d' % (filename, sent_num))

                word_num += 1
                id = 's' + str(sent_num) + 'w' + str(word_num)
                content = [id, st, et, word]
                mapping = ';'.join([str(x) for x in content])
                st = et
                et = round(et + time_gap, 6)

                fo.write(mapping + '\n')

        fo.close()


def main():
    # clean_pretrained_vec()

    force_alignment_STT()

main()