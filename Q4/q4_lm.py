import math
import pickle
from collections import Counter
from nltk import bigrams, trigrams
from nltk.corpus import brown

class AddKLM:
    def __init__(self, k=0.1):
        self.k = k
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.V = 0

    def fit(self, sentences):
        for sent in sentences:
            self.vocab.update(sent)
            self.unigrams.update(sent)
            self.bigrams.update(bigrams(sent))
            self.trigrams.update(trigrams(sent))
        self.V = max(1, len(self.vocab))

    def bigram_logprob(self, w1, w2):
        num = self.bigrams[(w1, w2)] + self.k
        den = self.unigrams[w1] + self.k * self.V
        return math.log(num / den)

    def trigram_logprob(self, w1, w2, w3):
        num = self.trigrams[(w1, w2, w3)] + self.k
        den = self.bigrams[(w1, w2)] + self.k * self.V
        return math.log(num / den)

    def sentence_logprob(self, tokens, order=3):
        words = [w.lower() for w in tokens if w]
        if order == 2:
            seq = ['<s>'] + words + ['</s>']
            return sum(self.bigram_logprob(a, b) for a, b in bigrams(seq))
        seq = ['<s>', '<s>'] + words + ['</s>']
        return sum(self.trigram_logprob(a, b, c) for a, b, c in trigrams(seq))

    def perplexity(self, tokens, order=3):
        n = max(1, len(tokens) + 1)
        return math.exp(-self.sentence_logprob(tokens, order) / n)


def build_brown_lms(k=0.1):
    sentences = []
    for sent in brown.sents():
        words = [w.lower() for w in sent if w.strip()]
        if words:
            sentences.append(['<s>', '<s>'] + words + ['</s>'])
    bi = AddKLM(k)
    tri = AddKLM(k)
    bi.fit(sentences)
    tri.fit(sentences)
    return bi, tri


def save_lms(bigram_lm, trigram_lm, path='q4_lms.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({'bigram': bigram_lm, 'trigram': trigram_lm}, f)


def load_lms(path='q4_lms.pkl'):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x['bigram'], x['trigram']
