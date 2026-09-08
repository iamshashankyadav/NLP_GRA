# Q4 Integrated Background Editor

This version implements all five parts of Q4 as one pipeline.

## 1. Prepare Q1 artifact once (if needed)

Q4 does not retrain Q1 during execution. `q1_artifacts.pkl` is already pre-generated. If you ever need to recreate it:

```bash
# From repository root
python Q4/prepare_q1_artifact.py

# Or from inside Q4
cd Q4
python prepare_q1_artifact.py
```

This creates the reusable trained Q1 segmentation LM + POS HMM artifact.

## 2. Run the integrated command-line pipeline

```bash
# From repository root
python Q4/run_q4_full.py

# Or from inside Q4
cd Q4
python run_q4_full.py
```

It loads Q1/Q3 artifacts, trains Q4's shared Brown add-k bigram/trigram models, induces the PTB PCFG, samples a random 6-sentence Gutenberg passage, simulates merged typing, produces live alerts, reports latency, and performs final PCFG/bigram/trigram analysis.

## 3. Run Streamlit (Deployment / Web UI)

```bash
# From repository root
streamlit run Q4/app.py

# Or from inside Q4
cd Q4
streamlit run app.py
```

The text area is processed incrementally when text is appended. If an earlier part of the text is edited, the checker resets and reprocesses the passage so stale state is not retained.

## 4. Speed Demon Benchmark

```bash
# From repository root
python Q4/benchmark.py

# Or from inside Q4
cd Q4
python benchmark.py
```

The benchmark uses exactly 1,000 Brown words and reports total/average segmentation+spelling latency and isolated grammar-trigger latency.

## Design choices

- merge probability p = 0.08: enough merged tokens to exercise Q1 without making nearly every boundary an error.
- grammar trigger N = 7: catches local problems without running the expensive grammar layer for every token.
- add-k smoothing k = 0.1 for Q4 Brown bigram/trigram models.
- Q1 Brown POS tags are mapped to PTB tags with `q4_tagmap.py` and supplied as lexical constraints to the CKY parser.
- PCFG is induced from the NLTK Penn Treebank sample after removing traces and functional labels and converting productions to CNF.
- Final method rule: prefer a successful PCFG parse with reasonable normalized log probability; otherwise use trigram for sentences of at least four words; otherwise use bigram.
- Real-word errors are checked at every grammar trigger using Q3's candidate generator and Q3 bigram probabilities.
