"""
t9_atypical_user.py — Test d'utilisateur atypique (T9).

Détecte les utilisateurs dont le volume d'écritures ou le montant moyen
est significativement éloigné de la médiane (z-score robuste basé sur le MAD).

Implémentation en SQL (DuckDB).
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def run_t9_atypical_user(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les utilisateurs atypiques.

    Args:
        df: Grand livre nettoyé.
        config: Section t9_atypical_user du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    z_threshold = config.get("z_threshold", 2.5)

    work = df[["entry_id", "debit", "credit", "user_id"]].copy()
    work["amount"] = work[["debit", "credit"]].max(axis=1)

    con = duckdb.connect()
    con.register("ledger", work)

    sql = f"""
    WITH user_stats AS (
        SELECT
            user_id,
            COUNT(DISTINCT entry_id) AS n_entries,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount
        FROM ledger
        GROUP BY user_id
    ),
    medians AS (
        SELECT
            MEDIAN(n_entries) AS median_entries,
            MEDIAN(avg_amount) AS median_avg_amount
        FROM user_stats
    ),
    global_stats AS (
        SELECT
            m.median_entries,
            m.median_avg_amount,
            MEDIAN(ABS(u.n_entries - m.median_entries)) AS mad_entries,
            MEDIAN(ABS(u.avg_amount - m.median_avg_amount)) AS mad_avg_amount
        FROM user_stats u
        CROSS JOIN medians m
        GROUP BY m.median_entries, m.median_avg_amount
    )
    SELECT
        u.user_id,
        u.n_entries,
        u.total_amount,
        u.avg_amount,
        g.median_entries,
        g.median_avg_amount,
        g.mad_entries,
        g.mad_avg_amount,
        CASE WHEN g.mad_entries > 0
             THEN ABS(u.n_entries - g.median_entries) / (1.4826 * g.mad_entries)
             ELSE 0
        END AS z_entries,
        CASE WHEN g.mad_avg_amount > 0
             THEN ABS(u.avg_amount - g.median_avg_amount) / (1.4826 * g.mad_avg_amount)
             ELSE 0
        END AS z_avg_amount
    FROM user_stats u
    CROSS JOIN global_stats g
    WHERE (
        CASE WHEN g.mad_entries > 0
             THEN ABS(u.n_entries - g.median_entries) / (1.4826 * g.mad_entries)
             ELSE 0
        END > {z_threshold}
    ) OR (
        CASE WHEN g.mad_avg_amount > 0
             THEN ABS(u.avg_amount - g.median_avg_amount) / (1.4826 * g.mad_avg_amount)
             ELSE 0
        END > {z_threshold}
    )
    ORDER BY z_entries + z_avg_amount DESC
    """


    result = con.execute(sql).fetchdf()
    con.close()

    results = []
    for _, row in result.iterrows():
        # Trouver toutes les écritures de cet utilisateur
        user_entries = df[df["user_id"] == row["user_id"]]["entry_id"].unique()

        reasons = []
        if row["z_entries"] > z_threshold:
            reasons.append(
                f"volume ({row['n_entries']} écritures, z={row['z_entries']:.2f})"
            )
        if row["z_avg_amount"] > z_threshold:
            reasons.append(
                f"montant moyen ({row['avg_amount']:.2f}€, z={row['z_avg_amount']:.2f})"
            )

        reason_str = " et ".join(reasons)

        for eid in user_entries:
            results.append({
                "entry_id": str(eid),
                "test_id": "T9",
                "observed_value": (
                    f"user={row['user_id']}, "
                    f"n_écritures={row['n_entries']}, "
                    f"montant_moyen={row['avg_amount']:.2f}€, "
                    f"z_volume={row['z_entries']:.2f}, "
                    f"z_montant={row['z_avg_amount']:.2f}"
                ),
                "threshold": f"z_seuil={z_threshold}",
                "message": (
                    f"Utilisateur atypique ({row['user_id']}) : {reason_str} — "
                    f"total {row['total_amount']:.2f}€ sur {row['n_entries']} écritures"
                ),
            })

    result_df = pd.DataFrame(results)
    logger.info("T9 Utilisateur atypique : %d signalements", len(result_df))
    return result_df
