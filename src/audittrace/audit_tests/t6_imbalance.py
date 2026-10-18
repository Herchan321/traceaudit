"""
t6_imbalance.py — Test de déséquilibre débit/crédit (T6).

Vérifie que la somme des débits est égale à la somme des crédits
pour chaque écriture (à une tolérance d'arrondi près).

Implémentation en Python.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_t6_imbalance(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les écritures déséquilibrées.

    Args:
        df: Grand livre nettoyé.
        config: Section t6_imbalance du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    tolerance = config.get("tolerance", 0.01)

    # Agréger par écriture
    entry_totals = df.groupby("entry_id").agg(
        total_debit=("debit", "sum"),
        total_credit=("credit", "sum"),
        n_lines=("line_id", "count"),
    ).reset_index()

    entry_totals["difference"] = abs(entry_totals["total_debit"] - entry_totals["total_credit"])
    entry_totals["difference"] = entry_totals["difference"].round(2)

    # Filtrer les écritures déséquilibrées
    imbalanced = entry_totals[entry_totals["difference"] > tolerance]

    results = []
    for _, row in imbalanced.iterrows():
        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T6",
            "observed_value": (
                f"total_débit={row['total_debit']:.2f}€, "
                f"total_crédit={row['total_credit']:.2f}€, "
                f"écart={row['difference']:.2f}€"
            ),
            "threshold": f"tolérance={tolerance}€",
            "message": (
                f"Écriture déséquilibrée : débits ({row['total_debit']:.2f}€) ≠ "
                f"crédits ({row['total_credit']:.2f}€), "
                f"écart de {row['difference']:.2f}€ "
                f"sur {row['n_lines']} lignes"
            ),
        })

    result_df = pd.DataFrame(results)
    logger.info("T6 Déséquilibre : %d signalements", len(result_df))
    return result_df
