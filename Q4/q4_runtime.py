import os
import sys
import nltk

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from q1_segmentation import load_q1_artifacts
from q3_spelling import load_q3_artifacts
from q4_lm import build_brown_lms
from q4_pcfg import induce_pcfg_from_treebank, CKYPCFG
from q4_pipeline import LiveChecker


def ensure_nltk_corpora():
    packages = ['brown', 'treebank', 'gutenberg', 'punkt', 'punkt_tab']
    for pkg in packages:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                nltk.download(pkg, quiet=True)



class Q4Runtime:
    def __init__(self, base_dir=None):
        base_dir = base_dir or _CURRENT_DIR
        self.base_dir = base_dir

        ensure_nltk_corpora()

        spell_candidates = [
            os.path.join(base_dir, 'spelling_corrector_artifacts.pkl'),
            os.path.join(base_dir, '..', 'Q3', 'part5_spelling_corrector_app', 'spelling_corrector_artifacts.pkl'),
            os.path.join(base_dir, '..', 'Q3', 'spelling_corrector_artifacts.pkl'),
        ]
        spell_path = None
        for p in spell_candidates:
            if os.path.exists(p):
                spell_path = p
                break

        q1_candidates = [
            os.path.join(base_dir, 'q1_artifacts.pkl'),
            os.path.join(base_dir, '..', 'Q1', 'q1_artifacts.pkl'),
        ]
        q1_path = None
        for p in q1_candidates:
            if os.path.exists(p):
                q1_path = p
                break

        if not q1_path:
            raise FileNotFoundError(
                'q1_artifacts.pkl is missing. Create it once using prepare_q1_artifact.py; Q4 itself does not retrain Q1.'
            )

        self.spell = load_q3_artifacts(spell_path if spell_path else 'spelling_corrector_artifacts.pkl')
        self.seg = load_q1_artifacts(q1_path, reference_vocab=self.spell.vocab)
        self.bigram, self.trigram = build_brown_lms(k=0.1)
        grammar, _ = induce_pcfg_from_treebank()
        self.parser = CKYPCFG(grammar)

    def checker(self, trigger_n=7):
        return LiveChecker(self.seg, self.spell, self.bigram, self.trigram, trigger_n)

