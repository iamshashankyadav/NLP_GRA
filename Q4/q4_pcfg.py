import math
import pickle
from collections import defaultdict
from nltk import Tree, Nonterminal
from nltk.corpus import treebank
from nltk.grammar import induce_pcfg


def clean_tree(tree):
    tree = tree.copy(deep=True)
    for node in tree.subtrees():
        label = node.label()
        if '-' in label and not label.startswith('-'):
            node.set_label(label.split('-')[0])
        if '=' in node.label():
            node.set_label(node.label().split('=')[0])
    def prune(node):
        if isinstance(node, str):
            return node
        children = []
        for child in node:
            if isinstance(child, Tree) and child.label() == '-NONE-':
                continue
            x = prune(child) if isinstance(child, Tree) else child
            if x is not None and (not isinstance(x, Tree) or len(x) > 0):
                children.append(x)
        node[:] = children
        return node if children else None
    return prune(tree)


def induce_pcfg_from_treebank():
    productions = []
    trees = []
    for raw in treebank.parsed_sents():
        t = clean_tree(raw)
        if t is None or len(t.leaves()) < 1:
            continue
        t.collapse_unary(collapsePOS=False)
        t.chomsky_normal_form(horzMarkov=2)
        productions.extend(t.productions())
        trees.append(t)
    grammar = induce_pcfg(Nonterminal('S'), productions)
    return grammar, trees


class CKYPCFG:
    def __init__(self, grammar):
        self.grammar = grammar
        self.lex = defaultdict(list)
        self.unary = defaultdict(list)
        self.binary = defaultdict(list)
        self.nonterminals = set(p.lhs() for p in grammar.productions())
        self.nonterminals.update(
            rhs for p in grammar.productions() for rhs in p.rhs() if isinstance(rhs, Nonterminal)
        )
        for p in grammar.productions():
            lhs = p.lhs()
            rhs = p.rhs()
            if len(rhs) == 1 and isinstance(rhs[0], str):
                self.lex[rhs[0].lower()].append((lhs, math.log(max(p.prob(), 1e-300))))
            elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
                self.unary[rhs[0]].append((lhs, math.log(max(p.prob(), 1e-300))))
            elif len(rhs) == 2:
                self.binary[(rhs[0], rhs[1])].append((lhs, math.log(max(p.prob(), 1e-300))))
        self._unary_closure_cache = {}

    def _unary_closure(self, chart_cell):
        changed = True
        while changed:
            changed = False
            items = list(chart_cell.items())
            for child, (score, tree) in items:
                for parent, lp in self.unary.get(child, []):
                    cand = (score + lp, Tree(str(parent), [tree]))
                    if parent not in chart_cell or cand[0] > chart_cell[parent][0]:
                        chart_cell[parent] = cand
                        changed = True

    def parse(self, tokens, forced_tags=None):
        tokens = [t.lower() for t in tokens]
        n = len(tokens)
        if not n:
            return None, None, 'no_parse'
        chart = [[{} for _ in range(n + 1)] for _ in range(n)]
        for i, word in enumerate(tokens):
            allowed = None
            if forced_tags and i < len(forced_tags) and forced_tags[i]:
                allowed = {forced_tags[i]}
            lexical = self.lex.get(word, [])
            for tag, lp in lexical:
                if allowed is not None and str(tag) not in allowed:
                    continue
                chart[i][i + 1][tag] = (lp, Tree(str(tag), [word]))
            self._unary_closure(chart[i][i + 1])
            if not chart[i][i + 1]:
                return None, None, 'no_coverage'

        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                cell = chart[i][j]
                for k in range(i + 1, j):
                    left = chart[i][k]
                    right = chart[k][j]
                    for b, (sb, tb) in left.items():
                        for c, (sc, tc) in right.items():
                            for a, rp in self.binary.get((b, c), []):
                                score = sb + sc + rp
                                old = cell.get(a)
                                if old is None or score > old[0]:
                                    cell[a] = (score, Tree(str(a), [tb, tc]))
                self._unary_closure(cell)

        start = Nonterminal('S')
        if start not in chart[0][n]:
            return None, None, 'no_parse'
        score, tree = chart[0][n][start]
        return tree, score, 'parsed'


def save_parser(grammar, path='q4_pcfg.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(grammar, f)


def load_parser(path='q4_pcfg.pkl'):
    with open(path, 'rb') as f:
        grammar = pickle.load(f)
    return CKYPCFG(grammar)

def parse_sentence(parser, tokens, forced_tags=None):
    return parser.parse(tokens, forced_tags=forced_tags)
