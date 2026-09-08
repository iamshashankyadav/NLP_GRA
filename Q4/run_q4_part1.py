import random
import sys

sys.path.insert(0, "./Q1")  # adjust if your Q1 folder lives elsewhere

from q1_segmentation import load_q1_artifacts
from q3_spelling import load_q3_artifacts
from q4_lm import build_brown_lms
from q4_pipeline import LiveChecker, run_simulated_passage, MERGE_PROB, GRAMMAR_TRIGGER_N


def sample_passage(corpus_name="gutenberg", n_sentences=6, seed=None):
    """Randomly sample a contiguous n_sentences-sentence passage."""
    import nltk
    rng = random.Random(seed)

    if corpus_name == "gutenberg":
        from nltk.corpus import gutenberg
        fileid = rng.choice(gutenberg.fileids())
        sents = gutenberg.sents(fileid)
    elif corpus_name == "brown":
        from nltk.corpus import brown
        fileid = rng.choice(brown.fileids())
        sents = brown.sents(fileid)
    elif corpus_name == "reuters":
        from nltk.corpus import reuters
        fileid = rng.choice(reuters.fileids())
        sents = reuters.sents(fileid)
    else:
        raise ValueError(corpus_name)

    if len(sents) <= n_sentences:
        start = 0
    else:
        start = rng.randint(0, len(sents) - n_sentences)
    chosen = sents[start:start + n_sentences]
    words = [w for sent in chosen for w in sent if w.isalpha()]
    return words, fileid


def main():
    print("Loading Q1 artifacts (segmentation LM + HMM tagger)...")
    q3_corrector_for_vocab = load_q3_artifacts("spelling_corrector_artifacts.pkl")
    seg_decoder = load_q1_artifacts("q1_artifacts.pkl", reference_vocab=q3_corrector_for_vocab.vocab)

    print("Loading Q3 spelling corrector artifacts...")
    spell_corrector = load_q3_artifacts("spelling_corrector_artifacts.pkl")

    print("Training Q4's shared Brown bigram/trigram LM (add-k smoothing)...")
    bigram_lm, trigram_lm = build_brown_lms(k=0.1)  # full Brown; drop sample_size

    print(f"\nMerge probability p = {MERGE_PROB}, grammar trigger N = {GRAMMAR_TRIGGER_N}\n")

    # Sample a passage (seed only for debugging, per assignment instructions)
    words, source_file = sample_passage("gutenberg", n_sentences=6, seed=None)
    print(f"Sampled passage from: {source_file}")
    print("Passage:", " ".join(words))
    print()

    checker = LiveChecker(seg_decoder, spell_corrector, bigram_lm, trigram_lm,
                           trigger_n=GRAMMAR_TRIGGER_N)
    state = run_simulated_passage(words, checker, delay=0.05, merge_prob=MERGE_PROB)

    print("\n=== FINAL RECONSTRUCTED TOKEN STREAM ===")
    print(" ".join(state.tokens_seen))

    print(f"\nSegmentation merges resolved: {state.seg_merges_resolved}")
    print(f"Spelling corrections applied: {state.spelling_corrections}")

    print("\n=== ALERTS ===")
    for a in state.alerts:
        print(f"[{a.kind}] pos={a.position}: {a.message}")

    print("\n=== LATENCY REPORT ===")
    report = checker.report_latency()
    print(f"Avg per-token segmentation+spelling check: {report['avg_seg_spell_ms']:.4f} ms")
    print(f"Avg per-trigger grammar/real-word check:   {report['avg_grammar_ms']:.4f} ms")


if __name__ == "__main__":
    main()
