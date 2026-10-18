"""
t2_duplicates.py — Test de doublons (T2).

Détecte les écritures dupliquées : même compte, même montant, même date
et même utilisateur dans une fenêtre de N jours.

Implémentation en SQL (DuckDB) pour démontrer la maîtrise SQL.
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def run_t2_duplicates(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les doublons avec DuckDB.

    Args:
        df: Grand livre nettoyé.
        config: Section t2_duplicates du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    window_days = config.get("window_days", 5)
    min_amount = config.get("min_amount", 0.01)

    # Préparer les données pour DuckDB
    work = df[
        ["entry_id", "line_id", "account", "debit", "credit", "posting_date", "user_id"]
    ].copy()
    work["amount"] = work[["debit", "credit"]].max(axis=1)

    con = duckdb.connect()
    con.register("ledger", work)

    sql = f"""
    WITH ranked AS (
        SELECT
            a.entry_id AS entry_id_a,
            b.entry_id AS entry_id_b,
            a.account,
            a.amount,
            a.posting_date AS date_a,
            b.posting_date AS date_b,
            a.user_id,
            ABS(CAST(a.posting_date AS DATE) - CAST(b.posting_date AS DATE)) AS day_diff
        FROM ledger a
        JOIN ledger b
            ON a.account = b.account
            AND a.amount = b.amount
            AND a.user_id = b.user_id
            AND a.entry_id < b.entry_id
            AND a.amount >= {min_amount}
        WHERE ABS(CAST(a.posting_date AS DATE) - CAST(b.posting_date AS DATE)) <= {window_days}
    )
    SELECT DISTINCT
        entry_id_b AS entry_id,
        entry_id_a AS original_entry_id,
        account,
        amount,
        date_a,
        date_b,
        user_id,
        day_diff
    FROM ranked
    ORDER BY entry_id_b
    """

    result = con.execute(sql).fetchdf()
    con.close()

    results = []
    for _, row in result.iterrows():
        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T2",
            "observed_value": (
                f"doublon de {row['original_entry_id']}, "
                f"compte={row['account']}, "
                f"montant={row['amount']}€, "
                f"écart={row['day_diff']}j"
            ),
            "threshold": f"fenêtre={window_days}j, montant_min={min_amount}€",
            "message": (
                f"Doublon potentiel : écriture identique à {row['original_entry_id']} "
                f"(compte {row['account']}, {row['amount']}€, "
                f"même utilisateur {row['user_id']}, "
                f"écart de {row['day_diff']} jour(s))"
            ),
        })

    result_df = pd.DataFrame(results)
    logger.info("T2 Doublons : %d signalements", len(result_df))
    return result_df
