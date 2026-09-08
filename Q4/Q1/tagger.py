import math
from collections import defaultdict, Counter

BEAM_WIDTH = 30
class HMMTagger:
    def __init__(self, tagged_sents):
        self.transition_counts = Counter()          # (t1, t2, t3)
        self.bigram_tag_counts = Counter()           # (t1, t2)
        self.emission_counts = defaultdict(Counter)  # tag -> word -> count
        self.tag_counts = Counter()                  # tag -> count
        self.word_possible_tags = defaultdict(set)   # word -> set of observed tags
        self.tags = set()
        self.vocab = set()
        self._train(tagged_sents)

    def _train(self, tagged_sents):
        for sent in tagged_sents:
            tags = ['<s>', '<s>'] + [t for w, t in sent] + ['</s>']
            for w, t in sent:
                w_lower = w.lower()
                self.emission_counts[t][w_lower] += 1
                self.tag_counts[t] += 1
                self.word_possible_tags[w_lower].add(t)
                self.tags.add(t)
                self.vocab.add(w_lower)

            for i in range(len(tags) - 2):
                t1, t2, t3 = tags[i], tags[i + 1], tags[i + 2]
                self.transition_counts[(t1, t2, t3)] += 1
                self.bigram_tag_counts[(t1, t2)] += 1

    def transition_prob(self, t1, t2, t3, k=0.01):
        num = self.transition_counts.get((t1, t2, t3), 0) + k
        denom = self.bigram_tag_counts.get((t1, t2), 0) + k * (len(self.tags) + 1)
        return num / denom

    def emission_prob(self, tag, word, k=0.01):
        word_lower = word.lower()
        num = self.emission_counts[tag].get(word_lower, 0) + k
        denom = self.tag_counts[tag] + k * (len(self.vocab) + 1)
        return num / denom

    def get_candidate_tags(self, word):
        w_lower = word.lower()
        if w_lower in self.word_possible_tags:
            return self.word_possible_tags[w_lower]
        return self.tags


def viterbi_tag(words, hmm, beam_width=BEAM_WIDTH):
    n = len(words)
    if n == 0:
        return []
    # dp[i] : { (t1, t2): (log_prob, backpointer_t0) }
    dp = [dict() for _ in range(n + 1)]
    dp[0][('<s>', '<s>')] = (0.0, None)

    for i in range(1, n + 1):
        word = words[i - 1]
        cand_tags = hmm.get_candidate_tags(word)

        for (t1, t2), (score, bp) in dp[i - 1].items():
            for t3 in cand_tags:
                trans = hmm.transition_prob(t1, t2, t3)
                emiss = hmm.emission_prob(t3, word)
                new_score = score + math.log(trans) + math.log(emiss)
                new_state = (t2, t3)
                if new_state not in dp[i] or new_score > dp[i][new_state][0]:
                    dp[i][new_state] = (new_score, t1)
        # Beam search pruning
        if len(dp[i]) > beam_width:
            top = sorted(dp[i].items(), key=lambda x: x[1][0], reverse=True)[:beam_width]
            dp[i] = dict(top)
    # Transition to end-of-sentence tag </s>
    best_score, best_state = -float('inf'), None
    for (t1, t2), (score, bp) in dp[n].items():
        total = score + math.log(hmm.transition_prob(t1, t2, '</s>'))
        if total > best_score:
            best_score, best_state = total, (t1, t2)
    if best_state is None:
        # Fallback if no valid state found
        return [(w, list(hmm.tags)[0]) for w in words]
    # Backtrack tags
    tags = []
    curr_state = best_state
    for i in range(n, 0, -1):
        score, prev_t = dp[i][curr_state]
        tags.append(curr_state[1])
        curr_state = (prev_t, curr_state[0])
    tags.reverse()
    return list(zip(words, tags))