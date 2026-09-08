import random
from collections import Counter
import pandas as pd


def char_offsets(words):
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w), w))
        pos += len(w)
    return offsets


def evaluate_pipeline(test_sents, segment_fn, tag_fn, n_samples=300, seed=42):
    random.seed(seed)
    sample = random.sample(test_sents, min(n_samples, len(test_sents)))

    seg_correct = 0
    seg_total = 0
    tag_correct = 0
    tag_total = 0
    confusion = Counter()  # (gold_tag, pred_tag) -> count
    seg_caused_errors = 0
    genuine_tag_errors = 0

    for sent in sample:
        gold_words = [w.lower() for w, t in sent]
        gold_tags = [t for w, t in sent]
        text = ''.join(gold_words)
        if not text:
            continue

        # 1. Segmentation
        pred_words = segment_fn(text)
        if isinstance(pred_words, tuple):
            pred_words = pred_words[0]

        # Span alignment
        gold_offsets = char_offsets(gold_words)
        pred_offsets = char_offsets(pred_words)

        gold_spans = {(s, e) for s, e, _ in gold_offsets}
        pred_spans = {(s, e) for s, e, _ in pred_offsets}

        seg_correct += len(gold_spans & pred_spans)
        seg_total += len(gold_spans)

        # 2. Tagging
        pred_tagged = tag_fn(pred_words)
        pred_tag_map = {(s, e): tag for (s, e, _), (_, tag) in zip(pred_offsets, pred_tagged)}
        gold_tag_map = {(s, e): tag for (s, e, _), tag in zip(gold_offsets, gold_tags)}

        for span, gold_t in gold_tag_map.items():
            tag_total += 1
            if span in pred_tag_map:
                # Word boundary was correctly segmented
                pred_t = pred_tag_map[span]
                if pred_t == gold_t:
                    tag_correct += 1
                else:
                    genuine_tag_errors += 1
                    confusion[(gold_t, pred_t)] += 1
            else:
                # Word boundary was wrong -> error is caused by segmentation mistake
                seg_caused_errors += 1

    seg_acc = seg_correct / seg_total if seg_total > 0 else 0.0
    tag_acc = tag_correct / tag_total if tag_total > 0 else 0.0

    return {
        'seg_accuracy': seg_acc,
        'tag_accuracy': tag_acc,
        'seg_correct': seg_correct,
        'seg_total': seg_total,
        'tag_correct': tag_correct,
        'tag_total': tag_total,
        'seg_caused_errors': seg_caused_errors,
        'genuine_tag_errors': genuine_tag_errors,
        'confusion_matrix': confusion,
    }


def evaluate_baselines(test_sents, greedy_seg_fn, mft_baseline, default_tag, n_samples=300, seed=42):
    """Evaluates greedy longest-match segmentation and Most Frequent Tag baseline."""
    random.seed(seed)
    sample = random.sample(test_sents, min(n_samples, len(test_sents)))

    seg_correct = 0
    seg_total = 0
    tag_correct = 0
    tag_total = 0

    for sent in sample:
        gold_words = [w.lower() for w, t in sent]
        gold_tags = [t for w, t in sent]
        text = ''.join(gold_words)
        if not text:
            continue

        pred_words = greedy_seg_fn(text)

        gold_offsets = char_offsets(gold_words)
        pred_offsets = char_offsets(pred_words)

        gold_spans = {(s, e) for s, e, _ in gold_offsets}
        pred_spans = {(s, e) for s, e, _ in pred_offsets}

        seg_correct += len(gold_spans & pred_spans)
        seg_total += len(gold_spans)

        pred_tagged = [(w, mft_baseline.get(w.lower(), default_tag)) for w in pred_words]
        pred_tag_map = {(s, e): tag for (s, e, _), (_, tag) in zip(pred_offsets, pred_tagged)}
        gold_tag_map = {(s, e): tag for (s, e, _), tag in zip(gold_offsets, gold_tags)}

        for span, gold_t in gold_tag_map.items():
            tag_total += 1
            if span in pred_tag_map and pred_tag_map[span] == gold_t:
                tag_correct += 1

    seg_acc = seg_correct / seg_total if seg_total > 0 else 0.0
    tag_acc = tag_correct / tag_total if tag_total > 0 else 0.0

    return {
        'seg_accuracy': seg_acc,
        'tag_accuracy': tag_acc,
        'seg_correct': seg_correct,
        'seg_total': seg_total,
        'tag_correct': tag_correct,
        'tag_total': tag_total,
    }


def confusion_matrix_to_df(confusion, tags, top_n=20):
    tag_counts = Counter()
    for (gold, pred), count in confusion.items():
        tag_counts[gold] += count
        tag_counts[pred] += count

    sorted_tags = [t for t, _ in tag_counts.most_common(top_n)] if top_n else sorted(list(tags))
    df = pd.DataFrame(0, index=sorted_tags, columns=sorted_tags)

    for (gold, pred), count in confusion.items():
        if gold in df.index and pred in df.columns:
            df.loc[gold, pred] += count

    return df
