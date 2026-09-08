import random
import re
import time
from dataclasses import dataclass, field

MERGE_PROB = 0.08
GRAMMAR_TRIGGER_N = 7
GRAMMAR_OUTLIER_FACTOR = 1.8
REAL_WORD_GAIN = 0.5

@dataclass
class Alert:
    kind: str
    position: int
    message: str

@dataclass
class PipelineState:
    tokens: list = field(default_factory=list)
    pos_tags: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    seg_merges: int = 0
    spelling_corrections: int = 0
    seg_merge_positions: set = field(default_factory=set)
    spelling_correction_positions: set = field(default_factory=set)
    seg_spell_ms: list = field(default_factory=list)
    grammar_ms: list = field(default_factory=list)
    trigger_results: list = field(default_factory=list)
    word_count: int = 0


def tokenize_text(text):
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[.!?,;:]", text)


def sample_words_to_stream(words, merge_prob=MERGE_PROB, rng=None):
    rng = rng or random.Random()
    stream = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and rng.random() < merge_prob:
            stream.append(words[i] + words[i + 1])
            i += 2
        else:
            stream.append(words[i])
            i += 1
    return stream


def simulate_merged_typing(tokens, merge_prob=MERGE_PROB, rng=None):
    rng = rng or random.Random()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", tok):
            if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", tokens[i + 1]) and rng.random() < merge_prob:
                out.append(tok + tokens[i + 1])
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


class LiveChecker:
    def __init__(self, seg_decoder, spell_corrector, grammar_bigram_lm,
                 grammar_trigram_lm, trigger_n=GRAMMAR_TRIGGER_N):
        self.seg = seg_decoder
        self.spell = spell_corrector
        self.bigram = grammar_bigram_lm
        self.trigram = grammar_trigram_lm
        self.trigger_n = trigger_n
        self.state = PipelineState()

    def _is_word(self, token):
        return bool(re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", token))

    def process_token(self, raw_token):
        if not raw_token:
            return
        if not self._is_word(raw_token):
            self.state.tokens.append(raw_token)
            self.state.pos_tags.append(None)
            return

        start = time.perf_counter()
        token = raw_token.lower()
        final_words = [token]
        final_tags = [None]

        suspicious = token not in self.spell.vocab or len(token) > self.seg.max_word_len
        if suspicious:
            words, tags, _, single_word_is_better = self.seg.segment_and_tag(token)
            if (not single_word_is_better) and len(words) > 1 and all(w in self.spell.vocab for w in words):
                final_words = words
                final_tags = tags
                self.state.seg_merges += 1
                self.state.seg_merge_positions.add(self.state.word_count)
                self.state.alerts.append(Alert(
                    'SEGMENT-ALERT', len(self.state.tokens),
                    f"{raw_token} -> {' '.join(words)} | POS={tags}"
                ))

        corrected = []
        corrected_tags = []
        for i, word in enumerate(final_words):
            if not self.spell.in_vocab(word):
                candidate, changed = self.spell.correct(word)
                if changed and candidate != word:
                    corrected.append(candidate)
                    corrected_tags.append(final_tags[i] if i < len(final_tags) else None)
                    self.state.spelling_corrections += 1
                    self.state.spelling_correction_positions.add(self.state.word_count + len(corrected) - 1)
                    self.state.alerts.append(Alert(
                        'SPELL-ALERT', len(self.state.tokens),
                        f"{word} -> {candidate}"
                    ))
                else:
                    corrected.append(word)
                    corrected_tags.append(final_tags[i] if i < len(final_tags) else None)
            else:
                corrected.append(word)
                corrected_tags.append(final_tags[i] if i < len(final_tags) else None)

        self.state.tokens.extend(corrected)
        self.state.pos_tags.extend(corrected_tags)
        self.state.word_count += len(corrected)
        self.state.seg_spell_ms.append((time.perf_counter() - start) * 1000)

        if self.state.word_count and self.state.word_count % self.trigger_n == 0:
            self._run_grammar_check()

    def _real_word_check(self, window):
        best = None
        for i, word in enumerate(window):
            if not self.spell.in_vocab(word):
                continue
            candidates = self.spell.generate_candidates(word)
            if not candidates:
                continue
            left = window[i - 1] if i else '<s>'
            right = window[i + 1] if i + 1 < len(window) else '</s>'
            actual = self.spell.bigram_prob(left, word) * self.spell.bigram_prob(word, right)
            for cand in candidates:
                trial = self.spell.bigram_prob(left, cand) * self.spell.bigram_prob(cand, right)
                if actual <= 0 or trial <= 0:
                    continue
                gain = __import__('math').log(trial / actual)
                if gain >= REAL_WORD_GAIN and (best is None or gain > best[0]):
                    best = (gain, word, cand)
        return best

    def _run_grammar_check(self):
        start = time.perf_counter()
        words = [x for x in self.state.tokens if self._is_word(x)]
        window = words[-self.trigger_n:]
        tri_ppl = self.trigram.perplexity(window, 3)
        bi_ppl = self.bigram.perplexity(window, 2)
        previous = [x['trigram_ppl'] for x in self.state.trigger_results]
        baseline = sum(previous) / len(previous) if previous else None
        implausible = tri_ppl > (baseline * GRAMMAR_OUTLIER_FACTOR) if baseline else False
        real_word = self._real_word_check(window)
        if implausible or real_word:
            parts = [f"window={' '.join(window)}", f"trigram_ppl={tri_ppl:.2f}", f"bigram_ppl={bi_ppl:.2f}"]
            if baseline:
                parts.append(f"baseline={baseline:.2f}")
            if real_word:
                gain, old, new = real_word
                parts.append(f"real-word candidate: {old} -> {new}, gain={gain:.2f}")
            self.state.alerts.append(Alert('GRAMMAR-ALERT', len(self.state.tokens), ' | '.join(parts)))
        self.state.trigger_results.append({'trigram_ppl': tri_ppl, 'bigram_ppl': bi_ppl, 'real_word': real_word is not None})
        self.state.grammar_ms.append((time.perf_counter() - start) * 1000)

    def report_latency(self):
        return {
            'avg_seg_spell_ms': sum(self.state.seg_spell_ms) / len(self.state.seg_spell_ms) if self.state.seg_spell_ms else 0.0,
            'avg_grammar_ms': sum(self.state.grammar_ms) / len(self.state.grammar_ms) if self.state.grammar_ms else 0.0,
            'seg_spell_total_ms': sum(self.state.seg_spell_ms),
            'grammar_total_ms': sum(self.state.grammar_ms),
        }


def run_simulated_passage(tokens, checker, delay=0.05, merge_prob=MERGE_PROB, rng=None):
    import time as _time
    for token in simulate_merged_typing(tokens, merge_prob, rng):
        checker.process_token(token)
        if delay:
            _time.sleep(delay)
    return checker.state
