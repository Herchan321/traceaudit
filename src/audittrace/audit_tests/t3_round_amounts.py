"""
t3_round_amounts.py — Test des montants ronds (T3).

Détecte les montants multiples d'une valeur seuil (ex. 1000€)
sur les comptes sensibles.

Implémentation en Python.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_t3_round_amounts(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les montants ronds sur comptes sensibles.

    Args:
        df: Grand livre nettoyé.
        config: Section t3_round_amounts du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    multiple = config.get("multiple", 1000)
    sensitive_prefixes = config.get("sensitive_accounts", ["6", "7"])

    work = df.copy()
    work["amount"] = work[["debit", "credit"]].max(axis=1)

    # Filtrer les comptes sensibles
    mask_sensitive = work["account"].apply(
        lambda a: any(str(a).startswith(str(p)) for p in sensitive_prefixes)
    )

    # Filtrer les montants ronds (multiples du seuil et > 0)
    mask_round = (work["amount"] > 0) & (work["amount"] % multiple == 0)

    flagged = work[mask_sensitive & mask_round]

    results = []
    for _, row in flagged.iterrows():
        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T3",
            "observed_value": f"montant={row['amount']}€, compte={row['account']}",
            "threshold": f"multiple_de={multiple}€, comptes_sensibles={sensitive_prefixes}",
            "message": (
                f"Montant rond : {row['amount']}€ (multiple de {multiple}€) "
                f"sur le compte sensible {row['account']} ({row['account_label']})"
            ),
        })

    result_df = pd.DataFrame(results)
    logger.info("T3 Montants ronds : %d signalements", len(result_df))
    return result_df
