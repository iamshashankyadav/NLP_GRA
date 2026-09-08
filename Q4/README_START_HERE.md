# Q4 — Where everything is and what order to run it in
```
Q4_package/
├── Q1/
│   ├── segmentation.py
│   ├── tagger.py
│   ├── data_handling.py
│   ├── morphology.py
│   ├── baselines.py
│   ├── evaluate.py
│   └── run_q1.py
├── q1_segmentation.py                wraps Q1's code for Q4 to use
├── q3_spelling.py                    wraps your Q3 pickle for Q4 to use
├── spelling_corrector_artifacts.pkl
├── q4_lm.py                          Part 3, shared bigram/trigram LM
├── q4_pcfg.py                        Part 2, PCFG + CKY parser
├── q4_tagmap.py                      Part 2, Brown->PTB tag mapping
├── q4_pipeline.py                    Part 1, live-typing checker
└── run_q4_part1.py                   script that runs everything together
```

Nothing here is a notebook — they're all plain `.py` files you run from a terminal.

## Step-by-step: what to actually do

**Step 1 — Train Q1 once, save it to a file.**
This is the only slow step (a few minutes). It only needs to happen ONE time.

Open a terminal, go into the `Q4_package` folder, then run:

```bash
cd Q4_package
Q1_DIR=./Q1 python3 q1_segmentation.py
```

What this does: loads the Brown corpus, trains Q1's `TrigramLM` and `HMMTagger`
(same as your `Q1/run_q1.py` does), then saves them into a new file called
`q1_artifacts.pkl` in the current folder. You'll see it print training
progress, then a handful of test-token results at the end, then finish.

**Step 2 — Check the file appeared.**

```bash
ls -la q1_artifacts.pkl
```

If it's there, you're done training. You never need to run Step 1 again
unless you delete this file.

**Step 3 — Run the actual Q4 Part 1 pipeline.**

```bash
python3 run_q4_part1.py
```

This samples a random passage from `nltk.corpus.gutenberg`, simulates typing
it (with some words randomly merged together), and prints out every
SEGMENT-ALERT, SPELL-ALERT, and GRAMMAR-ALERT as it "types" — plus a final
latency report. This is your Part 1 deliverable output. Run it a couple of
times (it samples a different passage each run) and save 2 of the printed
transcripts for your report, as the assignment asks for "at least two full
sample runs on different randomly sampled passages."

## If something errors

- `ModuleNotFoundError: No module named 'segmentation'` → you're not running
  from inside `Q4_package/`, or `Q1_DIR` isn't set correctly. Always run
  commands from inside the `Q4_package` folder.
- `FileNotFoundError: q1_artifacts.pkl` → you skipped Step 1, or you're
  running `run_q4_part1.py` from a different folder than where Step 1 saved
  the file.
- `conllu` import error inside `Q1/data_handling.py` → run
  `pip install conllu --break-system-packages` (or without that flag,
  depending on your Python setup) first.
- Anything else — copy the exact error message back to me.

## What's NOT built yet

Part 4 (final passage analysis / decision rule / summary table) and Part 5
(Streamlit deployment + Speed Demon benchmark) still need to be written.
Come back once Steps 1–3 above work for you, and we'll do those next.

## Bugfixes applied after the first real run (2nd version of this package)

Your first real run surfaced 3 genuine bugs, now fixed in `q1_segmentation.py`
and `q4_pipeline.py`:

1. **Merged words like "soonas" weren't being caught at all.** The
   segmentation adapter had backwards logic: if the DP found no good split,
   it was treated as "this token is fine as one word" instead of "this
   still needs checking." Fixed — now falls through to the spelling
   checker (and stays flagged) when this happens.
2. **Proper nouns were getting shredded** (e.g. "Musgroves" -> "m"+"us"+"groves")
   because Brown's training vocabulary barely covers names. Fixed with a
   deliberate rule: capitalized tokens are now skipped by segmentation/
   spelling entirely and passed through as-is. This is a real limitation
   worth mentioning in your report, not a hidden patch.
3. **The grammar alert fired on almost every single window** because the
   fixed perplexity threshold (5000) was far below what real prose scores
   under this LM (10,000-40,000+ is normal). Fixed by switching to an
   adaptive threshold — a window is now flagged only if its perplexity is
   notably higher than the passage's own recent average, not against a
   fixed global number. You'll want to justify the 1.8x multiplier
   (in `_run_grammar_check`) in your report the same way you'd justify N and p.

If you already ran Step 1 (Q1 training) with the old files, you do NOT need
to retrain — `q1_artifacts.pkl` is unaffected by these fixes. Just re-copy
the updated `q1_segmentation.py` and `q4_pipeline.py` into your working
folder and rerun `python run_q4_part1.py`.

## Part 2, Part 3, Part 4 — now added (`run_q4_full.py`)

`run_q4_part1.py` only demonstrates Part 1. To see Parts 1-4 running
TOGETHER on one passage, run:

```bash
python run_q4_full.py
```

This will:
1. Load Q1/Q3 artifacts, train Q4's Brown LM (Part 3), induce the PCFG
   from the full Penn Treebank sample (Part 2) — this step alone takes
   ~10-15 seconds.
2. Sample a passage and run it through Part 1's live-typing pipeline.
3. Take the corrected token stream and run Part 4's analysis: split into
   sentences, score each with PCFG + bigram + trigram, apply the decision
   rule, print the summary table.

### Important, honest findings from testing this end-to-end

**PCFG coverage on real passages is low — expect this, don't panic about it.**
When I tested this against real Gutenberg/KJV passages, the PCFG almost
never successfully parsed a sentence (0 out of 5 in one test run, 1 out of
5 in another). The reason: the Penn Treebank sample bundled with NLTK is
~1980s Wall Street Journal financial news, and its ~18,800 grammar
productions barely overlap with the vocabulary of literary prose (Austen,
KJV Bible, etc.) that `nltk.corpus.gutenberg` samples from. This is a
genuine, expected property of the method, not a bug — you should report it
directly:
- Report the "no_coverage" rate you observe across your sample runs.
- This is exactly why the assignment's own suggested decision rule treats
  PCFG as "use when it parses, else fall back" rather than assuming PCFG
  will usually succeed.

**Full-grammar PCFG parsing can also be slow.** With the full ~18,800
production grammar, some in-Treebank-domain sentences of 15-25 words took
5-8+ seconds to parse exhaustively (NLTK's `ViterbiParser` is not
beam-limited — it's exact CKY-style search, whose cost scales with
grammar size). `q4_pcfg.py` now uses `max_time=3.0` as a practical bound
for a live-typing context, and treats a timeout as its own graceful status
(`"timeout"`) rather than crashing or hanging. Report this as a design
decision with the actual timings you observe — don't just quietly set the
timeout and move on.

**Segmentation is still imperfect on real/rare words.** Rare or archaic
words (e.g. "spake" in KJV text) can still get wrongly fragmented by Q1's
DP segmenter if they weren't well-represented in its training data. This
is worth showing as an example in your Comparative Analysis / interaction
effects discussion, not hidden.

None of this means the code is broken — it means the assignment is testing
whether you can build a system that fails honestly and gracefully on hard
cases, and then discuss why. That discussion IS graded content (see the
Comparative Analysis marking criteria).


