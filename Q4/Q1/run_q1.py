import os
import sys
import pandas as pd
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_handling import (
    load_english_data, load_german_data, build_vocab, to_tagged_sents
)
from segmentation import TrigramLM, segment_viterbi
from tagger import HMMTagger, viterbi_tag
from morphology import (
    convert_english_to_morph, convert_german_to_morph
)
from baselines import (
    greedy_longest_match_segment, build_mft_baseline, tag_with_baseline
)
from evaluate import (
    evaluate_pipeline, evaluate_baselines, confusion_matrix_to_df
)

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def run_english(train_sents, dev_sents, test_sents, output_dir, n_samples=300):
    vocab_en, freq_en = build_vocab(train_sents)
    print(f"English Vocabulary Size: {len(vocab_en):,}")
    # 1. Train Trigram LM
    print("Training English Trigram Language Model...")
    lm_en = TrigramLM(train_sents, vocab_en)
    # 2. Train Standard POS Tagger (Brown fine-grained)
    print("Training English Trigram HMM Tagger (Standard)...")
    hmm_en = HMMTagger(train_sents)
    # 3. Train Morphology-Aware POS Tagger
    print("Training English Trigram HMM Tagger (Morphology-Aware)...")
    train_sents_morph = convert_english_to_morph(train_sents)
    test_sents_morph = convert_english_to_morph(test_sents)
    hmm_en_morph = HMMTagger(train_sents_morph)
    # 4. Train Baselines
    print("Building English Baselines...")
    mft_baseline_en, default_tag_en = build_mft_baseline(train_sents)
    # 5. Evaluate Main Model
    print(f"Evaluating English Proposed Pipeline on {n_samples} test sentences...")
    results_main = evaluate_pipeline(
        test_sents,
        lambda txt: segment_viterbi(txt, lm_en)[0],
        lambda words: viterbi_tag(words, hmm_en),
        n_samples=n_samples,
        seed=42
    )
    # 6. Evaluate Morphology-Aware Model
    print(f"Evaluating English Morphology-Aware Pipeline on {n_samples} test sentences...")
    results_morph = evaluate_pipeline(
        test_sents_morph,
        lambda txt: segment_viterbi(txt, lm_en)[0],
        lambda words: viterbi_tag(words, hmm_en_morph),
        n_samples=n_samples,
        seed=42
    )
    # 7. Evaluate Baselines
    print(f"Evaluating English Baselines on {n_samples} test sentences...")
    results_base = evaluate_baselines(
        test_sents,
        lambda txt: greedy_longest_match_segment(txt, vocab_en),
        mft_baseline_en,
        default_tag_en,
        n_samples=n_samples,
        seed=42
    )
    # Error analysis
    total_err = results_main['seg_caused_errors'] + results_main['genuine_tag_errors']
    pct_seg_err = (results_main['seg_caused_errors'] / total_err * 100) if total_err > 0 else 0
    pct_gen_err = (results_main['genuine_tag_errors'] / total_err * 100) if total_err > 0 else 0
    print("\n--- English Results ---")
    print(f"Segmentation Accuracy (Trigram+DP): {results_main['seg_accuracy']:.2%}")
    print(f"Segmentation Accuracy (Baseline):   {results_base['seg_accuracy']:.2%}")
    print(f"Segmentation Gain:                  {results_main['seg_accuracy'] - results_base['seg_accuracy']:+.2%}")
    print(f"Tagging Accuracy (HMM+Viterbi):     {results_main['tag_accuracy']:.2%}")
    print(f"Tagging Accuracy (Morph-Aware):     {results_morph['tag_accuracy']:.2%}")
    print(f"Tagging Accuracy (Baseline):        {results_base['tag_accuracy']:.2%}")
    print(f"Tagging Gain:                       {results_main['tag_accuracy'] - results_base['tag_accuracy']:+.2%}")
    print(f"Segmentation-Caused Errors:         {results_main['seg_caused_errors']} ({pct_seg_err:.1f}%)")
    print(f"Genuine Tagging Errors:             {results_main['genuine_tag_errors']} ({pct_gen_err:.1f}%)")
    # Confusion matrix
    cm_df = confusion_matrix_to_df(results_main['confusion_matrix'], hmm_en.tags, top_n=20)
    cm_path = os.path.join(output_dir, "english_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Saved English Confusion Matrix to: {cm_path}")
    # Write summary text
    with open(os.path.join(output_dir, "english_results.txt"), "w", encoding="utf-8") as f:
        f.write("=== ENGLISH EVALUATION RESULTS ===\n\n")
        f.write(f"Test Samples: {n_samples}\n")
        f.write(f"Trigram+DP Segmentation Accuracy: {results_main['seg_accuracy']:.4f}\n")
        f.write(f"Baseline Segmentation Accuracy:   {results_base['seg_accuracy']:.4f}\n")
        f.write(f"Segmentation Gain over Baseline:  {results_main['seg_accuracy'] - results_base['seg_accuracy']:+.4f}\n\n")
        f.write(f"HMM POS Tagging Accuracy:         {results_main['tag_accuracy']:.4f}\n")
        f.write(f"Morph-Aware Tagging Accuracy:     {results_morph['tag_accuracy']:.4f}\n")
        f.write(f"Baseline MFT Tagging Accuracy:    {results_base['tag_accuracy']:.4f}\n")
        f.write(f"Tagging Gain over Baseline:       {results_main['tag_accuracy'] - results_base['tag_accuracy']:+.4f}\n\n")
        f.write("Error-Source Breakdown:\n")
        f.write(f"- Segmentation-caused errors:     {results_main['seg_caused_errors']} ({pct_seg_err:.2f}%)\n")
        f.write(f"- Genuine tagging errors:         {results_main['genuine_tag_errors']} ({pct_gen_err:.2f}%)\n")

    return lm_en, hmm_en, hmm_en_morph, results_main, results_morph, results_base
def run_german(train_data_raw, dev_data_raw, test_data_raw, output_dir, n_samples=300):
    print("2. GERMAN EVALUATION")

    train_sents = to_tagged_sents(train_data_raw, use_morph=False)
    test_sents = to_tagged_sents(test_data_raw, use_morph=False)
    train_sents_morph = convert_german_to_morph(train_data_raw)
    test_sents_morph = convert_german_to_morph(test_data_raw)

    vocab_de, freq_de = build_vocab(train_sents)
    print(f"German Vocabulary Size: {len(vocab_de):,}")
    # 1. Train Trigram LM
    print("Training German Trigram Language Model...")
    lm_de = TrigramLM(train_sents, vocab_de)
    # 2. Train Standard POS Tagger (UPOS)
    print("Training German Trigram HMM Tagger (UPOS)...")
    hmm_de = HMMTagger(train_sents)
    # 3. Train Morphology-Aware POS Tagger (UPOS + Gender + Number)
    print("Training German Trigram HMM Tagger (Morphology-Aware)...")
    hmm_de_morph = HMMTagger(train_sents_morph)
    # 4. Train Baselines
    print("Building German Baselines...")
    mft_baseline_de, default_tag_de = build_mft_baseline(train_sents)
    # 5. Evaluate Main Model
    print(f"Evaluating German Proposed Pipeline on {n_samples} test sentences...")
    results_main = evaluate_pipeline(
        test_sents,
        lambda txt: segment_viterbi(txt, lm_de, max_word_len=25)[0],
        lambda words: viterbi_tag(words, hmm_de),
        n_samples=n_samples,
        seed=42
    )
    # 6. Evaluate Morphology-Aware Model
    print(f"Evaluating German Morphology-Aware Pipeline on {n_samples} test sentences...")
    results_morph = evaluate_pipeline(
        test_sents_morph,
        lambda txt: segment_viterbi(txt, lm_de, max_word_len=25)[0],
        lambda words: viterbi_tag(words, hmm_de_morph),
        n_samples=n_samples,
        seed=42
    )
    # 7. Evaluate Baselines
    print(f"Evaluating German Baselines on {n_samples} test sentences...")
    results_base = evaluate_baselines(
        test_sents,
        lambda txt: greedy_longest_match_segment(txt, vocab_de, max_word_len=25),
        mft_baseline_de,
        default_tag_de,
        n_samples=n_samples,
        seed=42
    )
    # Error analysis
    total_err = results_main['seg_caused_errors'] + results_main['genuine_tag_errors']
    pct_seg_err = (results_main['seg_caused_errors'] / total_err * 100) if total_err > 0 else 0
    pct_gen_err = (results_main['genuine_tag_errors'] / total_err * 100) if total_err > 0 else 0
    print("\n--- German Results ---")
    print(f"Segmentation Accuracy (Trigram+DP): {results_main['seg_accuracy']:.2%}")
    print(f"Segmentation Accuracy (Baseline):   {results_base['seg_accuracy']:.2%}")
    print(f"Segmentation Gain:                  {results_main['seg_accuracy'] - results_base['seg_accuracy']:+.2%}")
    print(f"Tagging Accuracy (HMM+Viterbi):     {results_main['tag_accuracy']:.2%}")
    print(f"Tagging Accuracy (Morph-Aware):     {results_morph['tag_accuracy']:.2%}")
    print(f"Tagging Accuracy (Baseline):        {results_base['tag_accuracy']:.2%}")
    print(f"Tagging Gain:                       {results_main['tag_accuracy'] - results_base['tag_accuracy']:+.2%}")
    print(f"Segmentation-Caused Errors:         {results_main['seg_caused_errors']} ({pct_seg_err:.1f}%)")
    print(f"Genuine Tagging Errors:             {results_main['genuine_tag_errors']} ({pct_gen_err:.1f}%)")
    # Confusion matrix
    cm_df = confusion_matrix_to_df(results_main['confusion_matrix'], hmm_de.tags, top_n=20)
    cm_path = os.path.join(output_dir, "german_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Saved German Confusion Matrix to: {cm_path}")
    # Write summary text
    with open(os.path.join(output_dir, "german_results.txt"), "w", encoding="utf-8") as f:
        f.write("=== GERMAN EVALUATION RESULTS ===\n\n")
        f.write(f"Test Samples: {n_samples}\n")
        f.write(f"Trigram+DP Segmentation Accuracy: {results_main['seg_accuracy']:.4f}\n")
        f.write(f"Baseline Segmentation Accuracy:   {results_base['seg_accuracy']:.4f}\n")
        f.write(f"Segmentation Gain over Baseline:  {results_main['seg_accuracy'] - results_base['seg_accuracy']:+.4f}\n\n")
        f.write(f"HMM POS Tagging Accuracy:         {results_main['tag_accuracy']:.4f}\n")
        f.write(f"Morph-Aware Tagging Accuracy:     {results_morph['tag_accuracy']:.4f}\n")
        f.write(f"Baseline MFT Tagging Accuracy:    {results_base['tag_accuracy']:.4f}\n")
        f.write(f"Tagging Gain over Baseline:       {results_main['tag_accuracy'] - results_base['tag_accuracy']:+.4f}\n\n")
        f.write("Error-Source Breakdown:\n")
        f.write(f"- Segmentation-caused errors:     {results_main['seg_caused_errors']} ({pct_seg_err:.2f}%)\n")
        f.write(f"- Genuine tagging errors:         {results_main['genuine_tag_errors']} ({pct_gen_err:.2f}%)\n")
    return lm_de, hmm_de, hmm_de_morph, results_main, results_morph, results_base

def run_sample_tests(lm_en, hmm_en, hmm_en_morph, lm_de, hmm_de, hmm_de_morph, output_dir):
    print("              3. SAMPLE TEST STRINGS                  ")

    test_cases_en = [
        ("thequickbrownfoxjumpsoverthelazydog", "Assignment English Benchmark"),
        ("naturallanguageprocessingisinteresting", "Additional English Sentence 1"),
        ("machinelearningmodelsevaluatedata", "Additional English Sentence 2"),
    ]
    test_cases_de = [
        ("autobahnmeistereiverwaltungsgebaeude", "Assignment German Compound Benchmark"),
        ("deraltebaeckerbacktbrot", "Additional German Sentence 1"),
        ("dieneuenschuhesindschwarz", "Additional German Sentence 2 (Agreement: Fem Pl)"),
    ]
    outputs = []
    outputs.append("=" * 60)
    outputs.append("ENGLISH SAMPLE TEST STRINGS")
    outputs.append("=" * 60)
    for raw_str, desc in test_cases_en:
        words, score = segment_viterbi(raw_str, lm_en)
        tagged_std = viterbi_tag(words, hmm_en)
        tagged_morph = viterbi_tag(words, hmm_en_morph)

        outputs.append(f"\n[{desc}]")
        outputs.append(f"Input:         {raw_str}")
        outputs.append(f"Segmented:     {words}")
        outputs.append(f"Standard POS:  {tagged_std}")
        outputs.append(f"Morph-Aware:   {tagged_morph}")
        print(f"\n[{desc}]\nInput: {raw_str}\nSegmented: {words}\nStandard: {tagged_std}\nMorph: {tagged_morph}")

    outputs.append("\n" + "=" * 60)
    outputs.append("GERMAN SAMPLE TEST STRINGS")
    outputs.append("=" * 60)
    for raw_str, desc in test_cases_de:
        words, score = segment_viterbi(raw_str, lm_de, max_word_len=25)
        tagged_std = viterbi_tag(words, hmm_de)
        tagged_morph = viterbi_tag(words, hmm_de_morph)

        outputs.append(f"\n[{desc}]")
        outputs.append(f"Input:         {raw_str}")
        outputs.append(f"Segmented:     {words}")
        outputs.append(f"Standard POS:  {tagged_std}")
        outputs.append(f"Morph-Aware:   {tagged_morph}")
        print(f"\n[{desc}]\nInput: {raw_str}\nSegmented: {words}\nStandard: {tagged_std}\nMorph: {tagged_morph}")

    out_path = os.path.join(output_dir, "sample_test_outputs.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
    print(f"\nSaved Sample Test Outputs to: {out_path}")
def save_summary_table(res_main_en, res_base_en, res_morph_en,
                       res_main_de, res_base_de, res_morph_de, output_dir):
    summary_df = pd.DataFrame({
        'Metric': [
            'Segmentation Accuracy (Trigram+DP)',
            'Segmentation Accuracy (Greedy Baseline)',
            'Segmentation Improvement (Delta)',
            'POS Tagging Accuracy (Trigram HMM)',
            'POS Tagging Accuracy (Morphology-Aware HMM)',
            'POS Tagging Accuracy (MFT Baseline)',
            'POS Tagging Improvement (Delta)',
            'Segmentation-Caused Errors (%)',
            'Genuine POS Tagging Errors (%)'
        ],
        'English (Brown)': [
            f"{res_main_en['seg_accuracy']:.2%}",
            f"{res_base_en['seg_accuracy']:.2%}",
            f"{(res_main_en['seg_accuracy'] - res_base_en['seg_accuracy']):+.2%}",
            f"{res_main_en['tag_accuracy']:.2%}",
            f"{res_morph_en['tag_accuracy']:.2%}",
            f"{res_base_en['tag_accuracy']:.2%}",
            f"{(res_main_en['tag_accuracy'] - res_base_en['tag_accuracy']):+.2%}",
            f"{(res_main_en['seg_caused_errors'] / max(1, res_main_en['seg_caused_errors'] + res_main_en['genuine_tag_errors'])):.1%}",
            f"{(res_main_en['genuine_tag_errors'] / max(1, res_main_en['seg_caused_errors'] + res_main_en['genuine_tag_errors'])):.1%}"
        ],
        'German (UD-GSD)': [
            f"{res_main_de['seg_accuracy']:.2%}",
            f"{res_base_de['seg_accuracy']:.2%}",
            f"{(res_main_de['seg_accuracy'] - res_base_de['seg_accuracy']):+.2%}",
            f"{res_main_de['tag_accuracy']:.2%}",
            f"{res_morph_de['tag_accuracy']:.2%}",
            f"{res_base_de['tag_accuracy']:.2%}",
            f"{(res_main_de['tag_accuracy'] - res_base_de['tag_accuracy']):+.2%}",
            f"{(res_main_de['seg_caused_errors'] / max(1, res_main_de['seg_caused_errors'] + res_main_de['genuine_tag_errors'])):.1%}",
            f"{(res_main_de['genuine_tag_errors'] / max(1, res_main_de['seg_caused_errors'] + res_main_de['genuine_tag_errors'])):.1%}"
        ]
    })
    csv_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print("            SIDE-BY-SIDE SUMMARY TABLE                 ")
    print(summary_df.to_string(index=False))
    print(f"\nSaved Side-by-Side Summary to: {csv_path}")
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    ensure_output_dir(output_dir)
    # 1. English
    train_en, dev_en, test_en = load_english_data(split_ratio=0.8)
    lm_en, hmm_en, hmm_en_morph, res_en, res_morph_en, base_en = run_english(
        train_en, dev_en, test_en, output_dir, n_samples=250
    )
    # 2. German
    german_path = os.path.join(os.path.dirname(base_dir), "UD_German-GSD")
    if not os.path.exists(german_path):
        german_path = "UD_German-GSD"

    train_de, dev_de, test_de = load_german_data(german_path)
    lm_de, hmm_de, hmm_de_morph, res_de, res_morph_de, base_de = run_german(
        train_de, dev_de, test_de, output_dir, n_samples=250
    )
    # 3. Sample benchmark tests
    run_sample_tests(lm_en, hmm_en, hmm_en_morph, lm_de, hmm_de, hmm_de_morph, output_dir)
    # 4. Summary table
    save_summary_table(res_en, base_en, res_morph_en, res_de, base_de, res_morph_de, output_dir)
if __name__ == "__main__":
    main()