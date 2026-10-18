"""
tools.py — Boîte à outils de l'Agent d'Audit AuditTrace.

Fournit des outils exécutables directement par l'agent LLM ou le moteur ReAct
pour interroger le grand livre (DuckDB), inspecter les pièces comptables,
analyser les résultats d'audit et consulter la réglementation ISA 240.
"""

from pathlib import Path

import duckdb

from audittrace.agent.knowledge import search_isa_norm

PROJECT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def get_ledger_path() -> Path:
    """Retourne le chemin vers le fichier grand livre nettoyé (parquet ou csv)."""
    parquet_path = OUTPUT_DIR / "general_ledger_clean.parquet"
    if parquet_path.exists():
        return parquet_path
    return OUTPUT_DIR / "general_ledger.csv"


def inspect_entry(entry_id: str) -> dict:
    """
    Inspecte en détail une écriture comptable spécifique.

    Args:
        entry_id: Identifiant de l'écriture (ex: 'JE-000123').

    Returns:
        Dictionnaire avec les lignes comptables, métadonnées, équilibre et alertes d'audit.
    """
    clean_id = entry_id.strip()
    ledger_path = str(get_ledger_path())

    con = duckdb.connect()
    # 1. Récupérer les lignes de l'écriture
    query = f"SELECT * FROM '{ledger_path}' WHERE entry_id = '{clean_id}' ORDER BY line_id"
    df_entry = con.execute(query).df()

    if df_entry.empty:
        return {"error": f"Écriture '{clean_id}' introuvable dans le grand livre."}

    lines = df_entry.to_dict(orient="records")
    total_debit = float(df_entry["debit"].sum())
    total_credit = float(df_entry["credit"].sum())
    imbalance = round(abs(total_debit - total_credit), 2)

    # 2. Récupérer les alertes des tests pour cette écriture
    alerts = []
    test_results_path = OUTPUT_DIR / "test_results.csv"
    if test_results_path.exists():
        tr_df = con.execute(
            f"SELECT test_id, observed_value, threshold, message "
            f"FROM '{test_results_path}' WHERE entry_id = '{clean_id}'"
        ).df()
        alerts = tr_df.to_dict(orient="records")

    return {
        "entry_id": clean_id,
        "n_lines": len(lines),
        "total_debit": round(total_debit, 2),
        "total_credit": round(total_credit, 2),
        "is_balanced": (imbalance == 0.0),
        "imbalance": imbalance,
        "document_date": str(lines[0].get("document_date")),
        "posting_date": str(lines[0].get("posting_date")),
        "created_at": str(lines[0].get("created_at")),
        "user_id": str(lines[0].get("user_id")),
        "source": str(lines[0].get("source")),
        "lines": lines,
        "audit_alerts": alerts,
    }


def query_ledger(sql_query: str) -> dict:
    """
    Exécute une requête SQL DuckDB sur le grand livre comptable.
    La table est accessible sous le nom 'ledger'.

    Exemple:
        SELECT account, SUM(debit) as total_debit FROM ledger GROUP BY account LIMIT 5

    Args:
        sql_query: Requête SQL à exécuter.
    """
    ledger_path = str(get_ledger_path()).replace("\\", "/")
    con = duckdb.connect()

    # Créer une vue 'ledger' pour faciliter les requêtes
    con.execute(f"CREATE VIEW ledger AS SELECT * FROM '{ledger_path}'")

    try:
        df = con.execute(sql_query).df()
        # Limiter à 50 lignes max pour ne pas saturer le contexte LLM
        truncated = len(df) > 50
        result_df = df.head(50)
        return {
            "row_count": len(df),
            "columns": list(result_df.columns),
            "rows": result_df.to_dict(orient="records"),
            "truncated": truncated,
        }
    except Exception as e:
        return {"error": f"Erreur SQL : {str(e)}"}


def get_audit_findings(risk_min: str = "moyen", limit: int = 15) -> dict:
    """
    Récupère les principales anomalies d'audit détectées, filtrées par niveau de risque.

    Args:
        risk_min: 'faible', 'moyen', ou 'élevé'.
        limit: Nombre maximum d'écritures à retourner.
    """
    test_results_path = OUTPUT_DIR / "test_results.csv"
    if not test_results_path.exists():
        return {"error": "Fichier test_results.csv introuvable. Exécutez 'audittrace run'."}

    con = duckdb.connect()
    # Grouper les tests par écriture
    summary_query = f"""
        SELECT
            entry_id,
            STRING_AGG(DISTINCT test_id, ', ') as tests_triggered,
            COUNT(DISTINCT test_id) as n_tests,
            STRING_AGG(message, ' | ') as details
        FROM '{test_results_path}'
        GROUP BY entry_id
        ORDER BY n_tests DESC
        LIMIT {int(limit)}
    """
    df = con.execute(summary_query).df()
    return {
        "count": len(df),
        "findings": df.to_dict(orient="records"),
    }


def calculate_financial_exposure(test_id: str | None = None) -> dict:
    """
    Calcule le montant financier total (en euros) concerné par les anomalies détectées.

    Args:
        test_id: Optionnel, filtre sur un test spécifique (ex: 'T2', 'T5', 'T6').
    """
    ledger_path = str(get_ledger_path()).replace("\\", "/")
    test_results_path = str(OUTPUT_DIR / "test_results.csv").replace("\\", "/")

    con = duckdb.connect()
    where_clause = f"WHERE tr.test_id = '{test_id.upper()}'" if test_id else ""

    query = f"""
        SELECT
            COUNT(DISTINCT l.entry_id) as affected_entries,
            ROUND(SUM(GREATEST(l.debit, l.credit)), 2) as total_volume_eur,
            ROUND(AVG(GREATEST(l.debit, l.credit)), 2) as avg_amount_eur,
            ROUND(MAX(GREATEST(l.debit, l.credit)), 2) as max_amount_eur
        FROM '{ledger_path}' l
        JOIN '{test_results_path}' tr ON l.entry_id = tr.entry_id
        {where_clause}
    """
    res = con.execute(query).df().to_dict(orient="records")[0]
    res["filter_test_id"] = test_id or "TOUS"
    return res


def search_isa240_standards(query: str) -> dict:
    """
    Recherche dans la base normative ISA 240 les clauses juridiques et professionnelles
    justifiant la détection ou recommandant les diligences d'audit.
    """
    matches = search_isa_norm(query, top_k=3)
    return {"query": query, "references": matches}


# Dictionnaire des outils pour l'agent
AUDIT_TOOLS = {
    "inspect_entry": {
        "function": inspect_entry,
        "description": "Inspecte les lignes, montants, dates et alertes d'une écriture (entry_id).",
        "parameters": {"entry_id": "string"},
    },
    "query_ledger": {
        "function": query_ledger,
        "description": "Exécute une requête SQL DuckDB sur la vue 'ledger'.",
        "parameters": {"sql_query": "string"},
    },
    "get_audit_findings": {
        "function": get_audit_findings,
        "description": "Liste les anomalies détectées triées par nombre de tests déclenchés.",
        "parameters": {"risk_min": "string (optionnel)", "limit": "integer (optionnel)"},
    },
    "calculate_financial_exposure": {
        "function": calculate_financial_exposure,
        "description": "Calcule l'impact financier en euros des écritures signalées.",
        "parameters": {"test_id": "string (optionnel: T1, T2, T5, T6, etc.)"},
    },
    "search_isa240_standards": {
        "function": search_isa240_standards,
        "description": "Recherche les clauses normatives ISA 240 justifiant les anomalies.",
        "parameters": {"query": "string"},
    },
}
