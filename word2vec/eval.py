# Find the nearest neighbors of given words
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm

import sys
sys.path.append(".")
from helper import WordVectorHelper

# Evaluation Parameters
eval_words = ['vier', 'august', 'automat', 'apotheker', 'haus', 'frankfurt', 'deutsche', 'großbritannien']


# Get nearest neighbors of words
def cos_sim(a, b):
    if a.shape != b.shape:
        return 0
    a = a / norm(a)
    b = b / norm(b)
    return dot(a, b)


def evaluate(words=eval_words, directory='', file=''):
    w2v_helper = WordVectorHelper(directory + file)
    id2word, word2id, embeddings, _ = w2v_helper.load_vec()

    top_k = 10

    for word in words:
        eval_id = word2id.get(word, None)
        if eval_id is None:
            print('Word "%s" not in dictionary!' % (word))
            continue

        sim = []
        for j in range(len(word2id)):
            cos = cos_sim(embeddings[eval_id], embeddings[j])
            sim.append(cos)

        nearest = np.argsort(sim)[::-1][1:top_k+1]
        # [print(sim[i]) for i in nearest]
        log_str = '"%s" nearest neighbors:' % word

        for k in range(top_k):
            log_str = '%s %s,' % (log_str, id2word[nearest[k]])

        print(log_str)


def main():
    dirname = os.path.dirname(__file__) + '/vec/'
    filename = '100.vec'

    evaluate(directory=dirname, file=filename)
