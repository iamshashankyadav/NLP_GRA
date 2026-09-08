from collections import defaultdict, Counter

MAX_WORD_LEN = 25


def greedy_longest_match_segment(text, vocab, max_word_len=MAX_WORD_LEN):

    text = text.lower()
    n = len(text)
    words = []
    i = 0
    while i < n:
        matched = False
        for j in range(min(n, i + max_word_len), i, -1):
            if text[i:j] in vocab:
                words.append(text[i:j])
                i = j
                matched = True
                break
        if not matched:
            # Consume 1 character if no vocab match found
            words.append(text[i:i + 1])
            i += 1
    return words


def build_mft_baseline(tagged_sents):

    word_tag_counts = defaultdict(Counter)
    overall_tag_counts = Counter()

    for sent in tagged_sents:
        for w, t in sent:
            w_lower = w.lower()
            word_tag_counts[w_lower][t] += 1
            overall_tag_counts[t] += 1

    mft_baseline = {
        word: counts.most_common(1)[0][0]
        for word, counts in word_tag_counts.items()
    }
    default_tag = overall_tag_counts.most_common(1)[0][0] if overall_tag_counts else 'NOUN'
    return mft_baseline, default_tag


def tag_with_baseline(words, mft_baseline, default_tag):
    return [(w, mft_baseline.get(w.lower(), default_tag)) for w in words]