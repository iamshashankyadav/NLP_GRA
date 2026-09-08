import os
from q1_segmentation import load_q1_artifacts
from q3_spelling import load_q3_artifacts
from q4_lm import build_brown_lms
from q4_pcfg import induce_pcfg_from_treebank, CKYPCFG
from q4_pipeline import LiveChecker

class Q4Runtime:
    def __init__(self, base_dir=None):
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        spell_path = os.path.join(base_dir, 'spelling_corrector_artifacts.pkl')
        q1_path = os.path.join(base_dir, 'q1_artifacts.pkl')
        if not os.path.exists(q1_path):
            raise FileNotFoundError('q1_artifacts.pkl is missing. Create it once using prepare_q1_artifact.py; Q4 itself does not retrain Q1.')
        self.spell = load_q3_artifacts(spell_path)
        self.seg = load_q1_artifacts(q1_path, reference_vocab=self.spell.vocab)
        self.bigram, self.trigram = build_brown_lms(k=0.1)
        grammar, _ = induce_pcfg_from_treebank()
        self.parser = CKYPCFG(grammar)

    def checker(self, trigger_n=7):
        return LiveChecker(self.seg, self.spell, self.bigram, self.trigram, trigger_n)
