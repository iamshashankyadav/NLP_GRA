from dataclasses import dataclass
from q4_pcfg import parse_sentence
from q4_tagmap import map_brown_to_ptb

@dataclass
class SentenceAnalysis:
    text: str
    pcfg_status: str
    pcfg_logprob: float | None
    bigram_logprob: float
    trigram_logprob: float
    chosen_method: str
    verdict: str
    n_seg_merges: int
    n_spelling_corrections: int
    parse_tree: str = ''


def split_sentences(tokens):
    sentences, current = [], []
    for token in tokens:
        if token in '.!?':
            if current:
                sentences.append(current)
                current = []
        elif token.isalpha() or token.replace("'", '').replace('’', '').isalpha():
            current.append(token)
    if current:
        sentences.append(current)
    return sentences


def method_decision(scores, n):
    if scores['pcfg_status'].startswith('parsed') and scores['pcfg_logprob'] is not None:
        avg = scores['pcfg_logprob'] / max(n, 1)
        if avg > -15:
            return 'pcfg'
    if n >= 4:
        return 'trigram'
    return 'bigram'


def verdict(scores, method, tri_threshold=-11.5, bi_threshold=-10.0):
    if method == 'pcfg':
        return 'plausible'
    if method == 'trigram':
        return 'plausible' if scores['trigram_logprob'] / max(scores['length'], 1) >= tri_threshold else 'implausible'
    return 'plausible' if scores['bigram_logprob'] / max(scores['length'], 1) >= bi_threshold else 'implausible'


def analyze_passage(tokens, pos_tags, parser, bigram_lm, trigram_lm,
                    seg_merge_positions=None, spelling_correction_positions=None):
    seg_merge_positions = seg_merge_positions or set()
    spelling_correction_positions = spelling_correction_positions or set()
    out = []
    cursor = 0
    for sent in split_sentences(tokens):
        n = len(sent)
        tags = pos_tags[cursor:cursor+n] if pos_tags else None
        mapped_tags = [map_brown_to_ptb(t) if t else None for t in tags] if tags else None

        tree, pcfg_lp, status = parser.parse(sent, forced_tags=mapped_tags)
        if status in ('no_coverage', 'no_parse'):
            fallback_tree, fallback_lp, fallback_status = parser.parse(sent, forced_tags=None)
            if fallback_status == 'parsed':
                tree, pcfg_lp, status = fallback_tree, fallback_lp, 'parsed_unconstrained_after_tag_conflict'

        bi_lp = bigram_lm.sentence_logprob(sent, 2)
        tri_lp = trigram_lm.sentence_logprob(sent, 3)
        scores = {
            'pcfg_status': status,
            'pcfg_logprob': pcfg_lp,
            'bigram_logprob': bi_lp,
            'trigram_logprob': tri_lp,
            'length': n
        }
        chosen = method_decision(scores, n)
        final = verdict(scores, chosen)
        tree_text = tree.pformat(margin=1000) if tree is not None else ''
        out.append(SentenceAnalysis(
            ' '.join(sent), status, pcfg_lp, bi_lp, tri_lp, chosen, final,
            sum(i in seg_merge_positions for i in range(cursor, cursor+n)),
            sum(i in spelling_correction_positions for i in range(cursor, cursor+n)),
            tree_text
        ))
        cursor += n
    return out


def print_summary_table(rows):
    headers = ['Sentence', 'PCFG', 'Bigram', 'Trigram', 'Method', 'Verdict', 'SegMerges', 'SpellFixes']
    print(' | '.join(headers))
    print('-' * 170)
    for i, r in enumerate(rows, 1):
        p = f'{r.pcfg_logprob:.3f}' if r.pcfg_logprob is not None else r.pcfg_status
        print(f'{i:02d}. {r.text[:65]:65} | {p:32} | {r.bigram_logprob:10.3f} | {r.trigram_logprob:10.3f} | {r.chosen_method:8} | {r.verdict:10} | {r.n_seg_merges:9} | {r.n_spelling_corrections:9}')
