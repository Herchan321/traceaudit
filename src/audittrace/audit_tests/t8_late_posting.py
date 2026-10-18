"""
t8_late_posting.py — Test de saisie tardive (T8).

Détecte les écritures dont l'écart entre la date du document et la date
de comptabilisation dépasse un seuil configurable.

Implémentation en Python.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_t8_late_posting(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les saisies tardives.

    Args:
        df: Grand livre nettoyé.
        config: Section t8_late_posting du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    max_days = config.get("max_days", 30)

    work = df.copy()
    work["posting_date"] = pd.to_datetime(work["posting_date"], errors="coerce")
    work["document_date"] = pd.to_datetime(work["document_date"], errors="coerce")
    work = work.dropna(subset=["posting_date", "document_date"])

    # Calculer l'écart en jours
    work["day_gap"] = (work["posting_date"] - work["document_date"]).dt.days

    # Filtrer les écritures tardives (écart positif supérieur au seuil)
    late = work[work["day_gap"] > max_days]

    # Dédupliquer par entry_id (prendre le max gap par écriture)
    late_entries = late.groupby("entry_id").agg(
        max_gap=("day_gap", "max"),
        posting_date=("posting_date", "first"),
        document_date=("document_date", "first"),
        user_id=("user_id", "first"),
    ).reset_index()

    results = []
    for _, row in late_entries.iterrows():
        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T8",
            "observed_value": (
                f"écart={row['max_gap']}j, "
                f"doc={row['document_date'].strftime('%Y-%m-%d')}, "
                f"compta={row['posting_date'].strftime('%Y-%m-%d')}"
            ),
            "threshold": f"max_jours={max_days}",
            "message": (
                f"Saisie tardive : {row['max_gap']} jours entre la date du "
                f"document ({row['document_date'].strftime('%Y-%m-%d')}) et la "
                f"comptabilisation ({row['posting_date'].strftime('%Y-%m-%d')}), "
                f"seuil : {max_days} jours, "
                f"utilisateur : {row['user_id']}"
            ),
        })

    result_df = pd.DataFrame(results)
    logger.info("T8 Saisie tardive : %d signalements", len(result_df))
    return result_df
