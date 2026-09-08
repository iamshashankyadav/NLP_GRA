import os
import streamlit as st
from q4_runtime import Q4Runtime
from q4_pipeline import tokenize_text, GRAMMAR_TRIGGER_N
from q4_analysis import analyze_passage

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_runtime():
    return Q4Runtime(BASE)

def process_incremental(text, runtime):
    tokens = tokenize_text(text)
    previous = st.session_state.get("processed_input", "")
    checker = st.session_state.get("checker")
    if checker is None or not text.startswith(previous):
        checker = runtime.checker(GRAMMAR_TRIGGER_N)
        start = 0
    else:
        start = len(tokenize_text(previous))
    for token in tokens[start:]:
        checker.process_token(token)
    st.session_state.checker = checker
    st.session_state.processed_input = text
    return checker

st.set_page_config(page_title="Q4 Integrated Background Editor", layout="wide")
st.title("Q4 Integrated Background Editor")
st.caption("Q1 segmentation/POS + Q3 spelling + Q4 grammar + PCFG + shared Brown n-gram analysis")

try:
    runtime = get_runtime()
except Exception as e:
    st.error(str(e))
    st.stop()

text = st.text_area("Type or paste a passage", height=240, placeholder="Start typing here...")
checker = process_incremental(text, runtime)

if text:
    st.subheader("Live alerts")
    if checker.state.alerts:
        for a in checker.state.alerts[-30:]:
            st.write(f"**[{a.kind}]** {a.message}")
    else:
        st.write("No alerts yet.")

    st.subheader("Corrected live stream")
    st.write(" ".join(checker.state.tokens))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segmentation merges", checker.state.seg_merges)
    c2.metric("Spelling corrections", checker.state.spelling_corrections)
    c3.metric("Grammar triggers", len(checker.state.trigger_results))
    c4.metric("Words processed", checker.state.word_count)

    latency = checker.report_latency()
    c1, c2 = st.columns(2)
    c1.metric("Avg segmentation + spelling (ms)", f"{latency['avg_seg_spell_ms']:.4f}")
    c2.metric("Avg grammar trigger (ms)", f"{latency['avg_grammar_ms']:.4f}")

    if st.button("Run final passage analysis"):
        rows = analyze_passage(
            checker.state.tokens, checker.state.pos_tags, runtime.parser,
            runtime.bigram, runtime.trigram,
            checker.state.seg_merge_positions,
            checker.state.spelling_correction_positions
        )
        st.subheader("Part 2 — PCFG constituency parsing")
        for i, row in enumerate(rows, 1):
            with st.expander(f"Sentence {i}: {row.text}"):
                st.write("Status:", row.pcfg_status)
                if row.parse_tree:
                    st.code(row.parse_tree)
                else:
                    st.write("No parse tree available.")

        st.subheader("Part 3 — Bigram / trigram language-model scores")
        st.dataframe([
            {"Sentence": i + 1, "Bigram log-probability": r.bigram_logprob, "Trigram log-probability": r.trigram_logprob}
            for i, r in enumerate(rows)
        ], use_container_width=True)

        st.subheader("Part 4 — Final sentence analysis")
        st.dataframe([r.__dict__ for r in rows], use_container_width=True)
