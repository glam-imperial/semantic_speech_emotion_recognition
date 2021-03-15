import collections
import matplotlib.pyplot as plt

from preprocess import preprocess_wiki_files


class Articles(object):
    def __init__(self, params=None):
        self.articles = preprocess_wiki_files()
        self.vocabulary_size = 0
        self.word2id = self.id2word = dict()
        self.count_wd = dict()
        self.params = params

    def get_average_sentence_len(self):
        l = []
        for a in self.articles:
            for s in a:
                l.append(len(s))

        print(sum(l) / len(l))

        c = collections.Counter(l)
        c = sorted(c.items())
        num = [i[0] for i in c]
        freq = [i[1] for i in c]

        plt.bar(num, freq)
        plt.xticks(range(len(freq)), num)
        plt.show()

    def build_dictionary(self):
        num_articles = len(self.articles)

        count_wd = dict()
        count_articles_with_wd = dict()

        for a in self.articles:
            appear = dict()

            for sentence in a:
                for word in sentence:
                    count_wd[word] = 1 if not count_wd.get(word, None) else (count_wd[word] + 1)
                    if appear.get(word, None) is None:
                        appear[word] = 1
                        count_articles_with_wd[word] = 1 if not count_articles_with_wd.get(word, None) \
                            else (count_articles_with_wd[word] + 1)
                    else:
                        appear[word] += 1

        print('Unique words: %d' % len(count_wd))

        # Remove words appear less than min occurences
        min_occurence = 7 if self.params is None else self.params.min_occurrence
        for w, c in list(count_wd.items()):
            if c < min_occurence:
                del count_wd[w]

        print('Vocab size after removing rare words: %d' % len(count_wd))

        # Remove words appear in more than n% of articles
        max_occurrence_percentage = 0.5 if self.params is None else self.params.max_occurrence_percentage
        for w, _ in list(count_wd.items()):
            percentage = count_articles_with_wd[w]/num_articles
            if percentage >= max_occurrence_percentage:
                del count_wd[w]

        print('Vocab size after removing popular words: %d' % len(count_wd))

        self.vocabulary_size = len(count_wd)
        count_wd = collections.OrderedDict(sorted(count_wd.items()))

        count = collections.Counter(count_wd).most_common(len(count_wd))
        print('Most common words:', count[:10])
        print('Least common words:', count[-10:])

        self.count_wd = count_wd

        # word2id, id2word
        id = 0
        for tp in count:
            w, _ = tp
            self.word2id[w] = id
            id += 1

        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        # Refine articles: remove words not in count_wd
        for a in self.articles:
            for idx in range(len(a)):
                s = a[idx]
                new_s = []
                for w in s:
                    if count_wd.get(w, None) is not None:
                        new_s.append(w)
                a[idx] = new_s

    def count_word_appearance(self, word):
        print('Word "%s" appears %d times' % (word, self.count_wd[word]))
        return self.count_wd[word]

    def get_vocab_size(self):
        return self.vocabulary_size

    def get_articles(self):
        return self.articles

    def get_word2id(self):
        return self.word2id

    def get_id2word(self):
        return self.id2word

    def get_count_word(self):
        return self.count_wd


def main():
    article = Articles(params=None)
    article.build_dictionary()
    article.count_word_appearance('haus')
    article.count_word_appearance('vier')
    article.count_word_appearance('august')
    article.count_word_appearance('automat')

