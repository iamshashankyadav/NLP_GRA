def brown_tag_to_morph_tag(tag):

    base_tag = tag.split('-')[0].split('+')[0].split('$')[0]
    is_poss = '$' in tag

    mapping = {
        # Nouns: Singular vs Plural
        'NN': 'NOUN-Sg',
        'NNS': 'NOUN-Pl',
        'NP': 'PROPN-Sg',
        'NPS': 'PROPN-Pl',
        'NR': 'NOUN-Loc-Sg',
        'NRS': 'NOUN-Loc-Pl',

        # Verbs: Tense, Aspect, Person/Number
        'VB': 'VERB-Base',
        'VBD': 'VERB-Past',
        'VBG': 'VERB-PresPart',
        'VBN': 'VERB-PastPart',
        'VBZ': 'VERB-3SgPres',
        'VBP': 'VERB-Non3Pres',
        'MD': 'VERB-Modal',
        'BE': 'AUX-Base',
        'BED': 'AUX-PastPl',
        'BEDZ': 'AUX-PastSg',
        'BEG': 'AUX-PresPart',
        'BEN': 'AUX-PastPart',
        'BEM': 'AUX-1SgPres',
        'BER': 'AUX-PresPl',
        'BEZ': 'AUX-3SgPres',
        'HV': 'AUX-HaveBase',
        'HVD': 'AUX-HavePast',
        'HVG': 'AUX-HavePresPart',
        'HVN': 'AUX-HavePastPart',
        'HVZ': 'AUX-Have3SgPres',
        'HVP': 'AUX-HaveNon3Pres',

        # Adjectives: Positive, Comparative, Superlative
        'JJ': 'ADJ-Pos',
        'JJR': 'ADJ-Comp',
        'JJS': 'ADJ-Super',
        'JJT': 'ADJ-SuperMorph',

        # Adverbs
        'RB': 'ADV-Pos',
        'RBR': 'ADV-Comp',
        'RBS': 'ADV-Super',

        # Pronouns: Person, Case, Number
        'PP': 'PRON-Personal',
        'PPO': 'PRON-Obj',
        'PPS': 'PRON-3SgSubj',
        'PPSS': 'PRON-Subj',
        'PN': 'PRON-Indef',

        # Determiners
        'DT': 'DET',
        'DTI': 'DET-Indef',
        'DTS': 'DET-Pl',
        'DTX': 'DET',
        'AT': 'DET-Art',

        # Prepositions / Conjunctions / Punctuation
        'IN': 'ADP',
        'CC': 'CCONJ',
        'CS': 'SCONJ',
        'CD': 'NUM',
        'UH': 'INTJ',
    }

    morph = mapping.get(base_tag, base_tag)
    if is_poss:
        morph += "-Poss"
    return morph


def convert_english_to_morph(tagged_sents):
    """Converts a dataset of English (word, brown_tag) sentences to morphology-aware tags."""
    return [[(w, brown_tag_to_morph_tag(t)) for w, t in sent] for sent in tagged_sents]


def convert_german_to_morph(conllu_data):
    morph_sents = []
    for sent in conllu_data:
        s = []
        for tok in sent:
            word = tok['word'].lower()
            feats = []
            if tok['gender']:
                feats.append(tok['gender'])  # Masc, Fem, Neut
            if tok['number']:
                feats.append(tok['number'])  # Sing, Plur
            tag = f"{tok['upos']}-{'-'.join(feats)}" if feats else tok['upos']
            s.append((word, tag))
        morph_sents.append(s)
    return morph_sents
