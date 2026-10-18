# AuditTrace

Automated accounting anomaly detection — **ISA 240** compliant.

AuditTrace is a Python toolkit that detects fraud and accounting anomalies in journal entries. It runs 9 statistical audit tests (T1–T9), scores each entry with a hybrid rule-based + Isolation Forest model, and provides an AI audit copilot that cites ISA 240 paragraphs.

## Features

| Test | Name | ISA 240 reference |
|------|------|-------------------|
| T1 | Benford's Law (MAD + z-score) | ISA 240.A43(5) |
| T2 | Duplicate entries | ISA 240.A43(7) |
| T3 | Round amounts | ISA 240.A43(5) |
| T4 | Unusual hours (night / weekend / holidays) | ISA 240.A43(4) |
| T5 | Splitting under approval thresholds (smurfing) | ISA 240.A43(6) |
| T6 | Debit / credit imbalance | ISA 240.A43(7) |
| T7 | Rare account pairs | ISA 240.A43(1) |
| T8 | Late posting | ISA 240.A43(3) |
| T9 | Atypical users (robust z-score) | ISA 240.A43(2) |

**Pipeline:** synthetic data generation → ingestion → quality checks → 9 tests → risk scoring → evaluation → Excel export.

**Interfaces:**
- CLI (`audittrace run | chat | memo`)
- FastAPI REST API
- Streamlit web UI
- Docker / docker-compose

## Architecture

```
src/audittrace/
├── __init__.py
├── cli.py              # Click CLI: run, chat, memo
├── api.py              # FastAPI REST API
├── app.py              # Streamlit web UI
├── ingest.py           # CSV import & column mapping
├── quality.py          # Data quality checks
├── generate_data.py    # Synthetic ledger + anomaly injection
├── scoring.py          # Rule score + Isolation Forest
├── evaluate.py         # Precision / recall / F1 vs ground truth
├── export.py           # Excel multi-sheet report
├── audit_tests/        # T1–T9 test modules
└── agent/              # Audit copilot (ReAct + ISA 240 RAG)
    ├── core.py         # ReAct engine, LLM + deterministic fallback
    ├── knowledge.py    # ISA 240 normative base
    ├── tools.py        # DuckDB, inspection, exposure, findings
    └── memo.py         # Audit memorandum generator
```

## Quick start

```bash
# 1. Install
pip install -e ".[rag,dev]"

# 2. Run the full pipeline (demo data)
audittrace run

# 3. Chat with the audit copilot
audittrace chat "What are the main anomalies?"

# 4. Generate the ISA 240 memorandum
audittrace memo

# 5. Start the Streamlit UI
streamlit run src/audittrace/app.py

# 6. Start the REST API
uvicorn audittrace.api:app --reload
```

## Streamlit UI

- **Home** — `Run demo audit` or `Upload my ledger` (CSV).
- **Overview** — entries audited, alerts, financial exposure, F1, risk score distribution.
- **Flagged entries** — filterable table; click an entry to see triggered tests, splitting twins, and the ISA 240 paragraph.
- **Audit copilot** — ReAct agent with visible tool → result → ISA citation trace.
- **Memo & reports** — download the memorandum (Markdown) and the Excel report.

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Ask the audit copilot |
| GET | `/findings` | List detected anomalies |
| POST | `/exposure` | Financial exposure (optional `test_id`) |
| GET | `/entry/{entry_id}` | Inspect a journal entry |
| POST | `/query` | SQL query on the ledger |
| GET | `/isa240` | Search ISA 240 clauses |
| POST | `/run` | Run the full pipeline |
| GET | `/memo` | Download the audit memorandum |

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Configuration

| File | Purpose |
|------|---------|
| `config/thresholds.yaml` | Per-test thresholds + data generation params |
| `config/weights.yaml` | Risk scoring weights + risk levels |
| `config/columns_mapping.json` | CSV column mapping |

## Requirements

- Python 3.11+
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `duckdb`, `pyarrow`, `xlsxwriter`, `pyyaml`, `rich`, `click`, `tabulate`
- Optional (`[rag]`): `langchain`, `langchain-community`, `chromadb`, `sentence-transformers`
- Optional (`[dev]`): `pytest`, `pytest-cov`, `ruff`

## License

MIT