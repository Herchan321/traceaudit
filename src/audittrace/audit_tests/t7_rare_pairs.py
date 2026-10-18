
"""
T7 — Détection des paires de comptes rares
dans les écritures comptables simples à deux lignes.
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def run_t7_rare_pairs(df, config):
    min_occurrences = int(config.get("min_occurrences", 3))

    sensitive_prefixes = tuple(
        str(prefix)
        for prefix in config.get("sensitive_accounts", ["6", "7", "1"])
    )

    columns = [
        "entry_id",
        "test_id",
        "observed_value",
        "threshold",
        "message",
    ]

    if df.empty or not sensitive_prefixes:
        return pd.DataFrame(columns=columns)

    con = duckdb.connect()
    con.register("ledger", df)

    try:
        # Ne conserver que les écritures comportant exactement
        # deux lignes : une ligne au débit et une ligne au crédit.
        pairs = con.execute("""
            WITH entry_sizes AS (
                SELECT
                    CAST(entry_id AS VARCHAR) AS entry_id,
                    COUNT(*) AS line_count
                FROM ledger
                GROUP BY entry_id
            ),
            debit_lines AS (
                SELECT
                    CAST(entry_id AS VARCHAR) AS entry_id,
                    CAST(account AS VARCHAR) AS debit_account
                FROM ledger
                WHERE debit > 0
            ),
            credit_lines AS (
                SELECT
                    CAST(entry_id AS VARCHAR) AS entry_id,
                    CAST(account AS VARCHAR) AS credit_account
                FROM ledger
                WHERE credit > 0
            )
            SELECT
                d.entry_id,
                d.debit_account,
                c.credit_account
            FROM debit_lines d
            JOIN credit_lines c USING (entry_id)
            JOIN entry_sizes s USING (entry_id)
            WHERE s.line_count = 2
        """).fetchdf()

    finally:
        con.close()

    if pairs.empty:
        return pd.DataFrame(columns=columns)

    pairs = pairs.drop_duplicates(
        subset=["entry_id", "debit_account", "credit_account"]
    )

    sensitive_mask = (
        pairs["debit_account"].astype(str).str.startswith(sensitive_prefixes)
        | pairs["credit_account"].astype(str).str.startswith(sensitive_prefixes)
    )

    pairs = pairs.loc[sensitive_mask].copy()

    if pairs.empty:
        return pd.DataFrame(columns=columns)

    # Fréquence de chaque paire dans les écritures simples.
    pairs["n_occurrences"] = pairs.groupby(
        ["debit_account", "credit_account"]
    )["entry_id"].transform("nunique")

    rare = pairs.loc[pairs["n_occurrences"] < min_occurrences]

    results = []

    for _, row in rare.iterrows():
        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T7",
            "observed_value": (
                f"{row['debit_account']} -> {row['credit_account']}"
            ),
            "threshold": min_occurrences,
            "message": (
                "Paire de comptes rare dans une écriture simple : "
                f"{row['debit_account']} -> {row['credit_account']} "
                f"({int(row['n_occurrences'])} occurrence(s))"
            ),
        })

    result_df = pd.DataFrame(results, columns=columns)

    logger.info("T7 Paires rares : %d signalements", len(result_df))

    return result_df
