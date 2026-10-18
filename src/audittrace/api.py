"""
api.py — API REST FastAPI pour AuditTrace.

Expose le pipeline d'audit, l'agent conversationnel ISA 240 et les
résultats d'analyse via des endpoints HTTP.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from audittrace.agent.core import AuditAgent
from audittrace.agent.memo import generate_audit_memorandum
from audittrace.agent.tools import (
    inspect_entry,
    query_ledger,
    get_audit_findings,
    calculate_financial_exposure,
    search_isa240_standards,
)
from audittrace.generate_data import generate_synthetic_data
from audittrace.ingest import ingest_csv
from audittrace.quality import run_quality_checks
from audittrace.scoring import compute_final_scores
from audittrace.evaluate import run_evaluation
from audittrace.export import export_to_excel
import yaml

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EXPECTED_COLUMNS = [
    "entry_id",
    "test_id",
    "observed_value",
    "threshold",
    "message",
]

app = FastAPI(
    title="AuditTrace API",
    description="API REST pour l'audit des écritures comptables (ISA 240)",
    version="1.0.0",
)

agent = AuditAgent()


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str


class RunResponse(BaseModel):
    status: str
    message: str
    details: dict[str, Any] | None = None


class ExposureRequest(BaseModel):
    test_id: str | None = None


class QueryRequest(BaseModel):
    sql_query: str


@app.get("/health", tags=["Système"])
def health():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "service": "audittrace"}


@app.post("/chat", response_model=ChatResponse, tags=["Agent ISA 240"])
def chat(req: ChatRequest):
    """Pose une question d'investigation à l'agent d'audit (ISA 240)."""
    try:
        response = agent.run(req.prompt)
        return ChatResponse(response=response)
    except Exception as e:
        logger.exception("Erreur agent chat")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/findings", tags=["Résultats"])
def findings(limit: int = 15):
    """Liste les anomalies détectées (nécessite un run préalable)."""
    try:
        result = get_audit_findings(limit=limit)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur findings")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exposure", tags=["Résultats"])
def exposure(req: ExposureRequest):
    """Calcule l'exposition financière des anomalies (optionnel par test)."""
    try:
        result = calculate_financial_exposure(test_id=req.test_id)
        return result
    except Exception as e:
        logger.exception("Erreur exposure")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entry/{entry_id}", tags=["Résultats"])
def entry(entry_id: str):
    """Inspecte une écriture comptable détaillée."""
    try:
        result = inspect_entry(entry_id=entry_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur entry")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", tags=["Requêtes"])
def query(req: QueryRequest):
    """Exécute une requête SQL DuckDB sur le grand livre."""
    try:
        result = query_ledger(sql_query=req.sql_query)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur query")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/isa240", tags=["Norme ISA 240"])
def isa240(query: str = "fraude écriture journal", top_k: int = 3):
    """Recherche les clauses normatives ISA 240 applicables."""
    try:
        matches = search_isa240_standards(query=query)
        return matches
    except Exception as e:
        logger.exception("Erreur isa240")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run", response_model=RunResponse, tags=["Pipeline"])
def run_pipeline():
    """Exécute le pipeline complet (génération → tests → export)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Génération données synthétiques
        ledger_path, ground_truth_path = generate_synthetic_data(
            CONFIG_DIR, OUTPUT_DIR
        )
        logger.info("Génération : %s", ledger_path)

        # 2. Ingestion
        df = ingest_csv(ledger_path, CONFIG_DIR)

        # 3. Qualité
        clean_df, quality_report = run_quality_checks(df, OUTPUT_DIR)

        # 4. Tests d'audit
        with open(CONFIG_DIR / "thresholds.yaml", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f) or {}

        tests = [
            ("t1_benford", __import__("audittrace.audit_tests.t1_benford", fromlist=["run_t1_benford"]).run_t1_benford),
            ("t2_duplicates", __import__("audittrace.audit_tests.t2_duplicates", fromlist=["run_t2_duplicates"]).run_t2_duplicates),
            ("t3_round_amounts", __import__("audittrace.audit_tests.t3_round_amounts", fromlist=["run_t3_round_amounts"]).run_t3_round_amounts),
            ("t4_unusual_hours", __import__("audittrace.audit_tests.t4_unusual_hours", fromlist=["run_t4_unusual_hours"]).run_t4_unusual_hours),
            ("t5_splitting", __import__("audittrace.audit_tests.t5_splitting", fromlist=["run_t5_splitting"]).run_t5_splitting),
            ("t6_imbalance", __import__("audittrace.audit_tests.t6_imbalance", fromlist=["run_t6_imbalance"]).run_t6_imbalance),
            ("t7_rare_pairs", __import__("audittrace.audit_tests.t7_rare_pairs", fromlist=["run_t7_rare_pairs"]).run_t7_rare_pairs),
            ("t8_late_posting", __import__("audittrace.audit_tests.t8_late_posting", fromlist=["run_t8_late_posting"]).run_t8_late_posting),
            ("t9_atypical_user", __import__("audittrace.audit_tests.t9_atypical_user", fromlist=["run_t9_atypical_user"]).run_t9_atypical_user),
        ]

        results = []
        counts = {}
        for test_name, test_function in tests:
            test_config = thresholds.get(test_name, {})
            result = test_function(clean_df, test_config)
            if result is not None and not result.empty:
                results.append(result)
            counts[test_name] = 0 if result is None else len(result)

        if results:
            test_results = pd.concat(results, ignore_index=True)
        else:
            test_results = pd.DataFrame(columns=EXPECTED_COLUMNS)

        for column in EXPECTED_COLUMNS:
            if column not in test_results.columns:
                test_results[column] = pd.NA
        test_results = test_results[EXPECTED_COLUMNS]
        test_results.to_csv(
            OUTPUT_DIR / "test_results.csv", index=False, encoding="utf-8-sig"
        )

        # 5. Scores + évaluation
        scores = compute_final_scores(clean_df, test_results, CONFIG_DIR)
        evaluation = run_evaluation(
            test_results, ground_truth_path, OUTPUT_DIR
        )

        # 6. Export Excel
        report_path = OUTPUT_DIR / "audittrace_report.xlsx"
        export_to_excel(
            scores,
            test_results,
            quality_report,
            evaluation,
            report_path,
        )

        return RunResponse(
            status="success",
            message="Pipeline terminé",
            details={
                "entries": int(len(clean_df)),
                "alerts": int(len(test_results)),
                "scored": int(len(scores)),
                "report": str(report_path),
                "counts": counts,
            },
        )
    except Exception as e:
        logger.exception("Erreur pipeline")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memo", tags=["Rapport"])
def memo():
    """Génère le Mémorandum d'Audit ISA 240 (fichier Markdown)."""
    try:
        path = generate_audit_memorandum()
        return FileResponse(
            path,
            media_type="text/markdown",
            filename="audit_memorandum_isa240.md",
        )
    except Exception as e:
        logger.exception("Erreur memo")
        raise HTTPException(status_code=500, detail=str(e))

