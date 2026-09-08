import pickle


class SpellingCorrector:
    def __init__(self, vocab, vocab_freq, bigram_counts, deletion_dict, vocab_size):
        self.vocab = vocab
        self.vocab_freq = vocab_freq
        self.bigram_counts = bigram_counts   # bigram_counts[w1][w2] -> count
        self.deletion_dict = deletion_dict
        self.vocab_size = vocab_size

    def in_vocab(self, word):
        return word.lower() in self.vocab

    # ---- Method A: deletion-neighbor candidate generation (Norvig-style) ----
    def _deletion_neighbors(self, word):
        """All strings formed by deleting exactly one character from word."""
        word = word.lower()
        return [word[:i] + word[i + 1:] for i in range(len(word))]

    def generate_candidates(self, word):
   
        word = word.lower()
        candidates = set()

        # Direction 1: word is missing letters relative to a vocab word
        # (e.g. user typed 'wrld', vocab has 'world' -> deleting 'o' from
        # 'world' gives 'wrld', so deletion_dict['wrld'] should contain 'world')
        if word in self.deletion_dict:
            candidates.update(self.deletion_dict[word])

        # Direction 2: word has an extra letter relative to a vocab word
        # (e.g. user typed 'onn', deleting one letter from 'onn' gives 'on',
        # which is itself in vocab, or maps to other vocab words via deletion_dict)
        for neighbor in self._deletion_neighbors(word):
            if neighbor in self.vocab:
                candidates.add(neighbor)
            if neighbor in self.deletion_dict:
                candidates.update(self.deletion_dict[neighbor])

        candidates.discard(word)
        return list(candidates)

    def correct(self, word):
        """Return (best_candidate, changed_bool). Picks the highest
        unigram-frequency candidate, as specified in the Q4 spec."""
        candidates = self.generate_candidates(word)
        if not candidates:
            return word, False
        best = max(candidates, key=lambda w: self.vocab_freq.get(w, 0))
        return best, True

    def bigram_count(self, w1, w2):
        return self.bigram_counts.get(w1, {}).get(w2, 0)

    def bigram_prob(self, w1, w2, k=0.1):
        """Add-k smoothed bigram probability using Q3's own bigram_counts,
        for the Part 1 real-word-error check (reuses Q3's bigram model,
        not Q4's separate Brown LM, per the assignment's reuse requirement)."""
        w1_total = sum(self.bigram_counts.get(w1, {}).values())
        count = self.bigram_count(w1, w2)
        return (count + k) / (w1_total + k * self.vocab_size)


def load_q3_artifacts(path="spelling_corrector_artifacts.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return SpellingCorrector(
        vocab=data["vocab"],
        vocab_freq=data["vocab_freq"],
        bigram_counts=data["bigram_counts"],
        deletion_dict=data["deletion_dict"],
        vocab_size=data["vocab_size"],
    )


if __name__ == "__main__":
    corrector = load_q3_artifacts("/mnt/user-data/uploads/spelling_corrector_artifacts.pkl")
    print(f"Loaded vocab of size {len(corrector.vocab)}")

    tests = ["onn", "helo", "wrld", "recieve", "teh"]
    for w in tests:
        in_vocab = corrector.in_vocab(w)
        cands = corrector.generate_candidates(w)
        best, changed = corrector.correct(w)
        print(f"'{w}': in_vocab={in_vocab}, candidates={cands[:8]}, correction='{best}' (changed={changed})")
