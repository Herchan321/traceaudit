"""
app.py — Interface Streamlit pour AuditTrace.

Parcours utilisateur :
1. Accueil : deux boutons — "Explore with demo data" ou "Upload my ledger".
2. Après analyse : onglets Overview, Flagged entries, Copilot, Memo & reports.
L'état (stage, résultats) est conservé dans st.session_state.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from audittrace.agent.core import AuditAgent
from audittrace.agent.memo import generate_audit_memorandum
from audittrace.agent.knowledge import search_isa_norm
from audittrace.agent.tools import inspect_entry, calculate_financial_exposure
from audittrace.ingest import ingest_csv
from audittrace.quality import run_quality_checks
from audittrace.scoring import compute_final_scores
from audittrace.evaluate import run_evaluation
from audittrace.export import export_to_excel
from audittrace.audit_tests.t1_benford import run_t1_benford
from audittrace.audit_tests.t2_duplicates import run_t2_duplicates
from audittrace.audit_tests.t3_round_amounts import run_t3_round_amounts
from audittrace.audit_tests.t4_unusual_hours import run_t4_unusual_hours
from audittrace.audit_tests.t5_splitting import run_t5_splitting
from audittrace.audit_tests.t6_imbalance import run_t6_imbalance
from audittrace.audit_tests.t7_rare_pairs import run_t7_rare_pairs
from audittrace.audit_tests.t8_late_posting import run_t8_late_posting
from audittrace.audit_tests.t9_atypical_user import run_t9_atypical_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TESTS = [
    ("t1_benford", run_t1_benford),
    ("t2_duplicates", run_t2_duplicates),
    ("t3_round_amounts", run_t3_round_amounts),
    ("t4_unusual_hours", run_t4_unusual_hours),
    ("t5_splitting", run_t5_splitting),
    ("t6_imbalance", run_t6_imbalance),
    ("t7_rare_pairs", run_t7_rare_pairs),
    ("t8_late_posting", run_t8_late_posting),
    ("t9_atypical_user", run_t9_atypical_user),
]

st.set_page_config(
    page_title="AuditTrace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Style sobre ──
st.markdown(
    """
    <style>
    .synthetic-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #1B2A4A;
        color: #FFFFFF;
        padding: 6px 16px;
        font-size: 0.85rem;
        text-align: center;
        z-index: 9999;
        letter-spacing: 0.2px;
    }
    .block-container {
        padding-top: 60px !important;
        padding-bottom: 2rem;
    }
    .home-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        background: #FAFAFA;
        text-align: center;
    }
    .home-card h3 {
        margin-top: 0;
        color: #1B2A4A;
    }
    .detail-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background: #FAFAFA;
        margin-top: 1rem;
    }
    .isa-ref {
        border-left: 3px solid #1B2A4A;
        padding-left: 10px;
        margin: 0.5rem 0;
        font-style: italic;
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="synthetic-banner">🧪 Synthetic dataset — generated for demonstration purposes</div>',
    unsafe_allow_html=True,
)

st.title("AuditTrace")
st.caption("Accounting anomaly detection — ISA 240")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "agent" not in st.session_state:
    st.session_state.agent = AuditAgent()
if "stage" not in st.session_state:
    st.session_state.stage = "home"
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "test_results" not in st.session_state:
    st.session_state.test_results = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def merge_entry_context(entry_id: str) -> dict:
    tr = st.session_state.test_results
    clean_df = st.session_state.clean_df
    entry_alerts = tr[tr["entry_id"] == entry_id] if tr is not None else pd.DataFrame()
    tests_triggered = list(entry_alerts["test_id"].unique()) if not entry_alerts.empty else []

    ctx = {
        "entry_id": entry_id,
        "tests_triggered": tests_triggered,
        "user_id": None,
        "document_date": None,
        "total_amount": 0.0,
    }
    if clean_df is not None:
        sub = clean_df[clean_df["entry_id"] == entry_id]
        if not sub.empty:
            ctx["user_id"] = str(sub.iloc[0]["user_id"])
            ctx["document_date"] = str(sub.iloc[0]["document_date"])
            ctx["total_amount"] = float(sub[["debit", "credit"]].max(axis=1).sum())
    return ctx


def find_splitting_twins(entry_id: str) -> list[str]:
    tr = st.session_state.test_results
    clean_df = st.session_state.clean_df
    if tr is None or clean_df is None:
        return []
    ctx = merge_entry_context(entry_id)
    if "T5" not in ctx["tests_triggered"]:
        return []
    uid = ctx["user_id"]
    doc_date = ctx["document_date"]
    t5_entries = tr[tr["test_id"] == "T5"]["entry_id"].unique()
    twins = []
    for eid in t5_entries:
        if eid == entry_id:
            continue
        sub = clean_df[clean_df["entry_id"] == eid]
        if sub.empty:
            continue
        if str(sub.iloc[0]["user_id"]) == uid and str(sub.iloc[0]["document_date"]) == doc_date:
            twins.append(eid)
    return twins


def run_pipeline(df: pd.DataFrame):
    with open(CONFIG_DIR / "thresholds.yaml", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f) or {}

    progress = st.progress(0, text="Starting audit pipeline…")

    # 1. Qualité
    progress.progress(10, text="Quality checks…")
    clean_df, quality_report = run_quality_checks(df, OUTPUT_DIR)

    # 2. Tests T1-T9
    results = []
    counts = {}
    n_tests = len(TESTS)
    for i, (test_name, test_function) in enumerate(TESTS):
        test_config = thresholds.get(test_name, {})
        result = test_function(clean_df, test_config)
        if result is not None and not result.empty:
            results.append(result)
        counts[test_name] = 0 if result is None else len(result)
        progress.progress(20 + int((i + 1) / n_tests * 60), text=f"Running {test_name.upper()}…")

    if results:
        test_results = pd.concat(results, ignore_index=True)
    else:
        test_results = pd.DataFrame(columns=["entry_id", "test_id", "observed_value", "threshold", "message"])

    expected = ["entry_id", "test_id", "observed_value", "threshold", "message"]
    for column in expected:
        if column not in test_results.columns:
            test_results[column] = pd.NA
    test_results = test_results[expected]
    test_results.to_csv(OUTPUT_DIR / "test_results.csv", index=False, encoding="utf-8-sig")

    # 3. Scores
    progress.progress(85, text="Computing risk scores…")
    scores = compute_final_scores(clean_df, test_results, CONFIG_DIR)

    # 4. Évaluation
    progress.progress(90, text="Running evaluation…")
    gt_path = OUTPUT_DIR / "ground_truth.csv"
    eval_result = run_evaluation(test_results, gt_path, OUTPUT_DIR)
    # run_evaluation retourne un dict ; on en extrait le DataFrame par test pour l'affichage
    if eval_result and eval_result.get("per_test"):
        evaluation = pd.DataFrame(eval_result["per_test"])
    else:
        evaluation = pd.DataFrame()

    # 5. Export Excel
    progress.progress(95, text="Exporting Excel report…")
    report_path = OUTPUT_DIR / "audittrace_report.xlsx"
    export_to_excel(scores, test_results, quality_report, eval_result, report_path)

    progress.progress(100, text="Done.")
    progress.empty()

    st.session_state.clean_df = clean_df
    st.session_state.test_results = test_results
    st.session_state.scores = scores
    st.session_state.quality_report = quality_report
    st.session_state.evaluation = evaluation
    st.session_state.stage = "results"

    return counts, report_path


# =====================================================================
# ACCUEIL
# =====================================================================
if st.session_state.stage == "home":
    st.markdown("### Get started")
    st.write("Choose how to load the general ledger and run the audit.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Explore with demo data")
        st.caption("Load the synthetic ledger (15 000 entries, injected anomalies T1–T9) and run the full audit in one click.")
        if st.button("Run demo audit", type="primary", use_container_width=True):
            from audittrace.generate_data import generate_synthetic_data
            ledger_path, _ = generate_synthetic_data(CONFIG_DIR, OUTPUT_DIR)
            df = ingest_csv(ledger_path, CONFIG_DIR)
            counts, report_path = run_pipeline(df)
            st.success(f"Demo audit completed — {counts.get('t6_imbalance', 0) + counts.get('t2_duplicates', 0)} anomalies detected.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Upload my ledger")
        st.caption("Upload your own general ledger CSV and run the audit on it.")
        uploaded = st.file_uploader("Select a CSV file", type=["csv"], key="home_csv")
        if uploaded is not None:
            save_path = OUTPUT_DIR / "grand_livre_upload.csv"
            save_path.write_bytes(uploaded.getbuffer())
            st.success(f"File uploaded: {uploaded.name}")
            if st.button("Run audit on this file", type="primary", use_container_width=True, key="run_upload"):
                df = ingest_csv(save_path, CONFIG_DIR)
                counts, report_path = run_pipeline(df)
                st.success("Audit completed on your data.")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# =====================================================================
# ESPACE DE RÉSULTATS (après analyse)
# =====================================================================

clean_df = st.session_state.clean_df
test_results = st.session_state.test_results
scores = st.session_state.scores
quality_report = st.session_state.quality_report
evaluation = st.session_state.evaluation

# Onglets de résultats
tab_overview, tab_flagged, tab_copilot, tab_reports = st.tabs(
    ["Overview", "Flagged entries", "Audit copilot", "Memo & reports"]
)

# ── Overview ──
with tab_overview:
    st.subheader("Overview")
    n_entries = int(len(clean_df)) if clean_df is not None else 0
    n_alerts = int(len(test_results)) if test_results is not None else 0
    exposure = calculate_financial_exposure()
    total_exposure = float(exposure.get("total_volume_eur", 0.0))
    f1_value = 0.0
    if evaluation is not None and not evaluation.empty and "f1_score" in evaluation.columns:
        f1_value = float(evaluation["f1_score"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries audited", f"{n_entries:,}")
    c2.metric("Alerts", f"{n_alerts:,}")
    c3.metric("Financial exposure", f"{total_exposure:,.2f} €")
    c4.metric("Avg F1 (synthetic)", f"{f1_value:.2%}")

    st.divider()
    st.subheader("Risk score distribution")
    if scores is not None and "risk_level" in scores.columns and not scores.empty:
        risk_counts = scores["risk_level"].value_counts()
        st.bar_chart(risk_counts)
    else:
        st.info("No risk scores available.")

# ── Flagged entries ──
with tab_flagged:
    st.subheader("Flagged entries")
    if test_results is None or test_results.empty:
        st.info("No flagged entries.")
    else:
        test_ids = sorted(test_results["test_id"].unique())
        selected_tests = st.multiselect("Filter by test", test_ids, default=test_ids)
        df_view = test_results[test_results["test_id"].isin(selected_tests)]
        st.dataframe(df_view, use_container_width=True, hide_index=True, height=420)

        st.divider()
        st.subheader("Inspect an entry")
        entry_ids = sorted(df_view["entry_id"].unique())
        selected_entry = st.selectbox("Entry ID", entry_ids)
        if selected_entry:
            ctx = merge_entry_context(selected_entry)
            twins = find_splitting_twins(selected_entry)
            isa_ref = None
            if ctx["tests_triggered"]:
                isa_results = search_isa_norm(", ".join(ctx["tests_triggered"]), top_k=1)
                if isa_results:
                    isa_ref = isa_results[0]
            detail = inspect_entry(selected_entry)

            st.markdown(
                f"""
                <div class="detail-card">
                <h4>Entry {selected_entry}</h4>
                <p>
                <strong>User:</strong> {ctx['user_id'] or '—'} &nbsp;|
                <strong>Document date:</strong> {ctx['document_date'] or '—'} &nbsp;|
                <strong>Total amount:</strong> {ctx['total_amount']:,.2f} € &nbsp;|
                <strong>Tests triggered:</strong> {', '.join(ctx['tests_triggered']) or '—'}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if twins:
                st.warning("🔗 Splitting twin(s) detected (same user, same date): " + ", ".join(f"`{t}`" for t in twins))
            if isa_ref:
                st.markdown(
                    f"""
                    <div class="isa-ref">
                    <strong>ISA 240 reference — {isa_ref['paragraph']}</strong><br>
                    {isa_ref['title']}<br>
                    « {isa_ref['content']} »
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if "error" not in detail:
                st.json(detail, expanded=False)

# ── Audit copilot ──
with tab_copilot:
    st.subheader("Audit copilot — ISA 240")
    st.caption("Ask investigative questions. The agent reasons step by step: tool → result → ISA paragraph.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Your audit question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Agent reasoning…", expanded=True) as status:
                response, steps = st.session_state.agent.run_with_trace(prompt)
                for i, step in enumerate(steps):
                    with st.expander(f"Step {i+1}: {step['tool']}", expanded=(i == len(steps) - 1)):
                        st.markdown("**Tool called:** `" + step["tool"] + "`")
                        st.markdown("**Arguments:**")
                        st.code(json.dumps(step["args"], ensure_ascii=False, indent=2))
                        st.markdown("**Result:**")
                        result = step["result"]
                        if isinstance(result, dict):
                            st.json(result, expanded=False)
                        else:
                            st.write(result)
                        if step.get("isa_paragraph"):
                            para = step["isa_paragraph"]
                            st.markdown("**ISA 240 paragraph cited:**")
                            st.markdown(
                                f"""
                                <div class="isa-ref">
                                <strong>{para.get('paragraph', '')}</strong> — {para.get('title', '')}<br>
                                « {para.get('content', '')} »
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                status.update(label="Reasoning complete", state="complete")

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ── Memo & reports ──
with tab_reports:
    st.subheader("Memo & reports")

    st.divider()
    st.subheader("Audit memorandum — ISA 240")
    st.caption("Formal memorandum for the engagement partner and audit committee.")
    if st.session_state.stage != "results":
        st.button("Generate memorandum", disabled=True)
    else:
        if st.button("Generate memorandum", type="primary"):
            with st.spinner("Generating ISA 240 memorandum…"):
                try:
                    path = generate_audit_memorandum()
                    content = path.read_text(encoding="utf-8")
                    st.download_button(
                        label="⬇️ Download memorandum (Markdown)",
                        data=content,
                        file_name="audit_memorandum_isa240.md",
                        mime="text/markdown",
                    )
                    st.markdown("### Preview")
                    st.markdown(content[:3000] + ("…" if len(content) > 3000 else ""))
                except Exception as e:
                    st.exception(e)

    st.divider()
    st.subheader("Download outputs")
    col1, col2 = st.columns(2)
    with col1:
        if test_results is not None:
            st.download_button(
                "⬇️ Test results (CSV)",
                data=test_results.to_csv(index=False).encode("utf-8"),
                file_name="test_results.csv",
                mime="text/csv",
                disabled=(st.session_state.stage != "results"),
            )
        if scores is not None:
            st.download_button(
                "⬇️ Risk scores (CSV)",
                data=scores.to_csv(index=False).encode("utf-8"),
                file_name="scores.csv",
                mime="text/csv",
                disabled=(st.session_state.stage != "results"),
            )
    with col2:
        if quality_report is not None and not quality_report.empty:
            st.download_button(
                "⬇️ Quality report (CSV)",
                data=quality_report.to_csv(index=False).encode("utf-8"),
                file_name="quality_report.csv",
                mime="text/csv",
                disabled=(st.session_state.stage != "results"),
            )
        xlsx_path = OUTPUT_DIR / "audittrace_report.xlsx"
        if xlsx_path.exists():
            st.download_button(
                "⬇️ Full report (Excel)",
                data=xlsx_path.read_bytes(),
                file_name="audittrace_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=(st.session_state.stage != "results"),
            )

st.sidebar.divider()
st.sidebar.caption("AuditTrace v1.0.0 — ISA 240")
