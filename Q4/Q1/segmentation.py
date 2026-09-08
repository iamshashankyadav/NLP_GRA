import math
from collections import Counter

MAX_WORD_LEN = 25
BEAM_WIDTH = 50

class TrigramLM:
    def __init__(self, tagged_sents, vocab, lambdas=(0.60, 0.25, 0.14, 0.01)):
        self.vocab = set(vocab)
        self.vocab_size = len(self.vocab) + 1  # +1 for <UNK>
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.total_words = 0
        self.l3, self.l2, self.l1, self.l0 = lambdas
        self._train(tagged_sents)

    def _train(self, tagged_sents):
        for sent in tagged_sents:
            words = ['<s>', '<s>'] + [w.lower() for w, t in sent] + ['</s>']
            self.total_words += len(words)
            for w in words:
                self.unigram_counts[w] += 1
            for i in range(len(words) - 1):
                self.bigram_counts[(words[i], words[i + 1])] += 1
            for i in range(len(words) - 2):
                self.trigram_counts[(words[i], words[i + 1], words[i + 2])] += 1

    def trigram_prob(self, w1, w2, w3):
        p_tri = (self.trigram_counts.get((w1, w2, w3), 0) / self.bigram_counts[(w1, w2)]
                 if (w1, w2) in self.bigram_counts else 0.0)
        p_bi = (self.bigram_counts.get((w2, w3), 0) / self.unigram_counts[w2]
                if w2 in self.unigram_counts else 0.0)
        p_uni = (self.unigram_counts.get(w3, 0) / self.total_words
                 if self.total_words > 0 else 0.0)
        p_unk = 1.0 / self.vocab_size
        return self.l3 * p_tri + self.l2 * p_bi + self.l1 * p_uni + self.l0 * p_unk


def segment_viterbi(text, lm, max_word_len=MAX_WORD_LEN, beam_width=BEAM_WIDTH):
    text = text.lower()
    n = len(text)
    if n == 0:
        return [], 0.0
    # dp[j] : { (w1, w2): (log_prob, (start_idx_i, prev_w1, prev_w2)) }
    dp = [dict() for _ in range(n + 1)]
    dp[0][('<s>', '<s>')] = (0.0, None)

    for j in range(1, n + 1):
        for i in range(max(0, j - max_word_len), j):
            candidate = text[i:j]
            # Valid word in vocabulary, or single character fallback to avoid dead ends
            if candidate in lm.vocab or (j - i == 1 and not dp[j]):
                for (w1, w2), (score, bp) in dp[i].items():
                    prob = lm.trigram_prob(w1, w2, candidate)
                    new_score = score + math.log(prob)
                    new_state = (w2, candidate)
                    if new_state not in dp[j] or new_score > dp[j][new_state][0]:
                        dp[j][new_state] = (new_score, (i, w1, w2))
        # Beam pruning to keep state space bounded and fast
        if len(dp[j]) > beam_width:
            top = sorted(dp[j].items(), key=lambda x: x[1][0], reverse=True)[:beam_width]
            dp[j] = dict(top)
    # Transition to end of sentence </s>
    best_score, best_state = -float('inf'), None
    for (w1, w2), (score, bp) in dp[n].items():
        total = score + math.log(lm.trigram_prob(w1, w2, '</s>'))
        if total > best_score:
            best_score, best_state = total, (w1, w2)
    if best_state is None:
        # Fallback if no valid state reached end
        return [text], -float('inf')
    # Backtrack path
    words = []
    curr_j, curr_state = n, best_state
    while curr_state is not None:
        if curr_state not in dp[curr_j]:
            break
        score, bp = dp[curr_j][curr_state]
        if bp is None:
            break
        i, w1_prev, w2_prev = bp
        words.append(text[i:curr_j])
        curr_j, curr_state = i, (w1_prev, w2_prev)

    words.reverse()
    return words, best_score