import os
import random
from collections import Counter
import nltk
from conllu import parse_incr

# Ensure reproducibility
SEED = 42
random.seed(SEED)


def load_english_data(split_ratio=0.8):

    try:
        from nltk.corpus import brown
        _ = brown.tagged_sents()
    except Exception:
        nltk.download('brown', quiet=True)
        from nltk.corpus import brown

    all_sents = list(brown.tagged_sents())
    # Shuffle with fixed seed for reproducibility
    shuffled = list(all_sents)
    random.Random(SEED).shuffle(shuffled)

    split_idx = int(split_ratio * len(shuffled))
    train_sents = shuffled[:split_idx]
    test_sents = shuffled[split_idx:]

    # Carve a 10% dev set from train
    dev_split = int(0.9 * len(train_sents))
    dev_sents = train_sents[dev_split:]
    train_sents = train_sents[:dev_split]

    print(f"[English Data] Train: {len(train_sents)} | Dev: {len(dev_sents)} | Test: {len(test_sents)}")
    return train_sents, dev_sents, test_sents


def get_words(tagged_sents):
    """Extracts all lowercase word tokens from tagged sentences."""
    return [w.lower() for sent in tagged_sents for w, t in sent]


def build_vocab(train_sents):
    """Builds a vocabulary set and word frequency counter from training sentences."""
    train_words = get_words(train_sents)
    vocab = set(train_words)
    word_freq = Counter(train_words)
    return vocab, word_freq


def load_conllu(filepath):
    sentences = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CoNLL-U file not found at: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            sent = []
            for token in tokenlist:
                if isinstance(token['id'], int):  # skip multiword-token ranges
                    feats = token.get('feats') or {}
                    sent.append({
                        'word': token['form'],
                        'lemma': token['lemma'],
                        'upos': token['upos'],
                        'xpos': token['xpos'],
                        'gender': feats.get('Gender', None),
                        'number': feats.get('Number', None),
                        'case': feats.get('Case', None),
                        'person': feats.get('Person', None),
                        'tense': feats.get('Tense', None),
                    })
            if sent:
                sentences.append(sent)
    return sentences


def to_tagged_sents(data, use_morph=False):
    tagged = []
    for sent in data:
        s = []
        for tok in sent:
            word = tok['word'].lower()
            if use_morph:
                feats = []
                if tok['gender']:
                    feats.append(tok['gender'][:4])  # Fem / Masc / Neut
                if tok['number']:
                    feats.append(tok['number'][:2])  # Si / Pl
                tag = f"{tok['upos']}-{'-'.join(feats)}" if feats else tok['upos']
            else:
                tag = tok['upos']
            s.append((word, tag))
        tagged.append(s)
    return tagged


def load_german_data(dataset_dir="UD_German-GSD"):
    train_path = os.path.join(dataset_dir, "de_gsd-ud-train.conllu")
    dev_path = os.path.join(dataset_dir, "de_gsd-ud-dev.conllu")
    test_path = os.path.join(dataset_dir, "de_gsd-ud-test.conllu")

    train_data = load_conllu(train_path)
    dev_data = load_conllu(dev_path)
    test_data = load_conllu(test_path)

    print(f"[German Data] Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")
    return train_data, dev_data, test_data


if __name__ == "__main__":
    train_en, dev_en, test_en = load_english_data()
    vocab_en, freq_en = build_vocab(train_en)
    print(f"English Vocab Size: {len(vocab_en)}")

    if os.path.exists("UD_German-GSD"):
        train_de_raw, dev_de_raw, test_de_raw = load_german_data()
        train_de = to_tagged_sents(train_de_raw, use_morph=False)
        vocab_de, freq_de = build_vocab(train_de)
        print(f"German Vocab Size: {len(vocab_de)}")
