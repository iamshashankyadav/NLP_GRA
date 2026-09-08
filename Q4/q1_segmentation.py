import pickle
import sys
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_Q1_DIRS = [
    os.environ.get("Q1_DIR"),
    os.path.join(_CURRENT_DIR, "..", "Q1"),
    os.path.join(_CURRENT_DIR, "Q1"),
    "./Q1"
]
_Q1_DIR = None
for _cand in _CANDIDATE_Q1_DIRS:
    if _cand and os.path.isdir(_cand):
        _Q1_DIR = os.path.abspath(_cand)
        break

if _Q1_DIR and _Q1_DIR not in sys.path:
    sys.path.insert(0, _Q1_DIR)

from segmentation import TrigramLM, segment_viterbi, MAX_WORD_LEN, BEAM_WIDTH as SEG_BEAM_WIDTH
from tagger import HMMTagger, viterbi_tag, BEAM_WIDTH as TAG_BEAM_WIDTH



class SegmentationDecoder:

    def __init__(self, lm, hmm, max_word_len=MAX_WORD_LEN,
                 seg_beam_width=SEG_BEAM_WIDTH, tag_beam_width=TAG_BEAM_WIDTH,
                 reference_vocab=None):
        self.lm = lm
        self.hmm = hmm
        self.max_word_len = max_word_len
        self.seg_beam_width = seg_beam_width
        self.tag_beam_width = tag_beam_width
        # Falls back to the LM's own vocab if no external reference is given
        self.reference_vocab = reference_vocab if reference_vocab is not None else lm.vocab

    def segment_and_tag(self, token):
     
        token_lower = token.lower()

        # Run the DP segmenter restricted to this token
        split_words, split_score = segment_viterbi(
            token_lower, self.lm,
            max_word_len=self.max_word_len,
            beam_width=self.seg_beam_width,
        )
        single_word_in_vocab = token_lower in self.reference_vocab

        single_word_is_better = single_word_in_vocab

        if single_word_is_better:
            tagged = viterbi_tag([token_lower], self.hmm, beam_width=self.tag_beam_width)
            words = [w for w, t in tagged]
            tags = [t for w, t in tagged]
            return words, tags, split_score, True

        tagged = viterbi_tag(split_words, self.hmm, beam_width=self.tag_beam_width)
        words = [w for w, t in tagged]
        tags = [t for w, t in tagged]
        return words, tags, split_score, False


def train_and_save_q1_artifacts(train_sents, output_path="q1_artifacts.pkl"):

    from data_handling import build_vocab

    vocab, freq = build_vocab(train_sents)
    print(f"Training TrigramLM on {len(train_sents)} sentences (vocab={len(vocab)})...")
    lm = TrigramLM(train_sents, vocab)
    print("Training HMMTagger...")
    hmm = HMMTagger(train_sents)

    with open(output_path, "wb") as f:
        pickle.dump({"lm": lm, "hmm": hmm}, f)
    print(f"Saved Q1 artifacts to {output_path}")
    return lm, hmm


def load_q1_artifacts(path="q1_artifacts.pkl", reference_vocab=None):
    if not os.path.isabs(path):
        candidates = [
            path,
            os.path.join(_CURRENT_DIR, path),
            os.path.join(_CURRENT_DIR, "..", "Q1", path)
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    with open(path, "rb") as f:
        data = pickle.load(f)
    return SegmentationDecoder(data["lm"], data["hmm"], reference_vocab=reference_vocab)


if __name__ == "__main__":
    from data_handling import load_english_data

    out_path = os.path.join(_CURRENT_DIR, "q1_artifacts.pkl")
    train_sents, dev_sents, test_sents = load_english_data(split_ratio=0.8)
    lm, hmm = train_and_save_q1_artifacts(train_sents, out_path)

    decoder = SegmentationDecoder(lm, hmm)

    # Smoke test: a merged token and a legitimate long word
    for tok in ["andthe", "thequickbrownfox", "extraordinary", "cats"]:
        words, tags, score, single_better = decoder.segment_and_tag(tok)
        print(f"'{tok}' -> words={words} tags={tags} score={score:.2f} single_word_is_better={single_better}")

