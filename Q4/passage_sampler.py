import random
import re


def sample_passage(n_sentences=6, seed=None):
    from nltk.corpus import gutenberg
    rng = random.Random(seed)
    fileid = rng.choice(gutenberg.fileids())
    sents = gutenberg.sents(fileid)
    if len(sents) <= n_sentences:
        start = 0
    else:
        start = rng.randint(0, len(sents) - n_sentences)
    return sents[start:start + n_sentences], fileid


def normalize_sentence(sent):
    out = []
    for w in sent:
        if re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", w):
            out.append(w.lower())
        elif w in '.!?':
            out.append(w)
    return out


def passage_to_tokens(sentences):
    tokens = []
    for sent in sentences:
        tokens.extend(normalize_sentence(sent))
        if not tokens or tokens[-1] not in '.!?':
            tokens.append('.')
    return tokens
