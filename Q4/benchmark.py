import os
import random
import time
from q4_runtime import Q4Runtime
from q4_pipeline import simulate_merged_typing

BASE = os.path.dirname(os.path.abspath(__file__))


def get_1000_words():
    from nltk.corpus import brown
    words = [w.lower() for w in brown.words() if w.isalpha()]
    if len(words) < 1000:
        raise RuntimeError('Brown corpus does not contain enough words')
    return words[:1000]


def main():
    rt = Q4Runtime(BASE)
    clean = get_1000_words()
    stream = simulate_merged_typing(clean, 0.08, random.Random(42))

    checker = rt.checker(7)
    t0 = time.perf_counter()
    for token in stream:
        checker.process_token(token)
    total_live = (time.perf_counter() - t0) * 1000

    grammar = rt.checker(7)
    words_added = 0
    t0 = time.perf_counter()
    for token in clean:
        grammar.state.tokens.append(token)
        grammar.state.pos_tags.append(None)
        words_added += 1
        grammar.state.word_count += 1
        if words_added % grammar.trigger_n == 0:
            grammar._run_grammar_check()
    grammar_total = (time.perf_counter() - t0) * 1000

    print('SPEED DEMON')
    print('Exactly 1000 source words:', len(clean))
    print('Incoming tokens after simulated merges:', len(stream))
    print(f'Segmentation + spelling total: {total_live:.3f} ms')
    print(f'Segmentation + spelling average per incoming token: {total_live / len(stream):.6f} ms')
    print(f'Grammar-only total: {grammar_total:.3f} ms')
    print(f'Grammar-only average per trigger: {grammar_total / max(1, len(grammar.state.grammar_ms)):.6f} ms')

if __name__ == '__main__':
    main()
