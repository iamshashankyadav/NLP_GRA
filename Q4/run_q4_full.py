import os
import sys
import random
from q4_runtime import Q4Runtime
from passage_sampler import sample_passage, passage_to_tokens
from q4_pipeline import MERGE_PROB, GRAMMAR_TRIGGER_N, run_simulated_passage
from q4_analysis import analyze_passage, print_summary_table

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'Q1'))


def main():
    print('Loading Q1/Q3 artifacts and Q4 models...')
    runtime = Q4Runtime(BASE)
    sentences, source = sample_passage(6)
    tokens = passage_to_tokens(sentences)

    print('Source:', source)
    print('Merge probability:', MERGE_PROB)
    print('Grammar trigger:', GRAMMAR_TRIGGER_N)
    print('Sampled sentences:', len(sentences))
    print('Passage:')
    print(' '.join(tokens))

    print('\nPART 1 - LIVE BACKGROUND CHECKER')
    checker = runtime.checker(GRAMMAR_TRIGGER_N)
    state = run_simulated_passage(tokens, checker, delay=0.0, merge_prob=MERGE_PROB, rng=random.Random())
    print('\nALERTS')
    for a in state.alerts:
        print(f'[{a.kind}] {a.message}')
    print('\nPART 1 LATENCY')
    print(checker.report_latency())
    print('Segmentation merges:', state.seg_merges)
    print('Spelling corrections:', state.spelling_corrections)
    print('Grammar triggers:', len(state.trigger_results))

    print('\nPART 2 - PCFG CONSTITUENCY PARSING')
    print('PCFG trained from Penn Treebank and converted to CNF.')
    print('Brown POS tags are mapped to PTB tags before constrained parsing.')

    print('\nPART 3 - SHARED SMOOTHED N-GRAM MODELS')
    print('Brown add-k bigram and trigram models trained with k=0.1.')
    print('Both models are used for sentence-level scoring and grammar triggers.')
    print('Sentence scoring will be shown below for every sampled sentence.')

    rows = analyze_passage(
        state.tokens,
        state.pos_tags,
        runtime.parser,
        runtime.bigram,
        runtime.trigram,
        state.seg_merge_positions,
        state.spelling_correction_positions
    )

    print('\nPART 4 - FINAL PASSAGE ANALYSIS')
    print('Sentences analyzed:', len(rows))
    print_summary_table(rows)

    print('\nPCFG PARSE RESULTS')
    for i, row in enumerate(rows, 1):
        print(f'\nSentence {i}: {row.text}')
        print('Status:', row.pcfg_status)
        if row.parse_tree:
            print(row.parse_tree)
        else:
            print('No parse tree available.')

    print('\nFINAL METHOD COUNTS')
    counts = {}
    for row in rows:
        counts[row.chosen_method] = counts.get(row.chosen_method, 0) + 1
    print(counts)

if __name__ == '__main__':
    main()
