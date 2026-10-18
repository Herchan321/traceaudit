"""
t5_splitting.py — Test de fractionnement (T5).

Détecte les écritures fractionnées : plusieurs écritures du même utilisateur,
même compte, même jour, chacune juste sous le seuil d'approbation mais dont
la somme le dépasse.

Implémentation en SQL (DuckDB).
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def run_t5_splitting(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les fractionnements sous seuil d'approbation.

    Args:
        df: Grand livre nettoyé.
        config: Section t5_splitting du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    threshold = config.get("approval_threshold", 10000)
    margin_pct = config.get("margin_pct", 0.10)
    min_entries = config.get("min_entries", 2)

    lower_bound = threshold * (1 - margin_pct)

    work = df[["entry_id", "account", "debit", "credit", "posting_date", "user_id"]].copy()
    work["amount"] = work[["debit", "credit"]].max(axis=1)

    con = duckdb.connect()
    con.register("ledger", work)

    sql = f"""
    WITH entry_amounts AS (
        -- Montant total par écriture (somme des débits)
        SELECT
            entry_id,
            account,
            user_id,
            CAST(posting_date AS DATE) AS posting_day,
            SUM(debit) AS total_debit
        FROM ledger
        WHERE debit > 0
        GROUP BY entry_id, account, user_id, CAST(posting_date AS DATE)
    ),
    potential_splits AS (
        -- Groupes d'écritures du même utilisateur, même compte, même jour
        -- où chaque écriture est sous le seuil mais le total dépasse
        SELECT
            user_id,
            account,
            posting_day,
            COUNT(DISTINCT entry_id) AS n_entries,
            SUM(total_debit) AS group_total,
            MIN(total_debit) AS min_amount,
            MAX(total_debit) AS max_amount,
            LIST(entry_id) AS entry_ids
        FROM entry_amounts
        WHERE total_debit > 0
            AND total_debit < {threshold}
            AND total_debit >= {lower_bound}
        GROUP BY user_id, account, posting_day
        HAVING COUNT(DISTINCT entry_id) >= {min_entries}
            AND SUM(total_debit) > {threshold}
    )
    SELECT * FROM potential_splits
    ORDER BY group_total DESC
    """

    result = con.execute(sql).fetchdf()
    con.close()

    results = []
    for _, row in result.iterrows():
        entry_ids = row["entry_ids"]
        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]

        for eid in entry_ids:
            results.append({
                "entry_id": str(eid),
                "test_id": "T5",
                "observed_value": (
                    f"total_groupe={row['group_total']:.2f}€, "
                    f"n_écritures={row['n_entries']}, "
                    f"plage=[{row['min_amount']:.2f}€-{row['max_amount']:.2f}€]"
                ),
                "threshold": (
                    f"seuil_approbation={threshold}€, "
                    f"marge={margin_pct:.0%}, "
                    f"borne_basse={lower_bound:.2f}€"
                ),
                "message": (
                    f"Fractionnement potentiel : {row['n_entries']} écritures du même "
                    f"utilisateur ({row['user_id']}), même compte ({row['account']}), "
                    f"même jour ({row['posting_day']}), "
                    f"chacune sous {threshold}€ mais total {row['group_total']:.2f}€"
                ),
            })

    result_df = pd.DataFrame(results)
    logger.info("T5 Fractionnement : %d signalements", len(result_df))
    return result_df
