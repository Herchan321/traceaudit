"""
t1_benford.py — Test de la loi de Benford (T1).

Vérifie que la distribution du premier chiffre significatif des montants
suit la loi de Benford, groupé par préfixe de compte.
Utilise le MAD (Mean Absolute Deviation) avec les seuils de Nigrini et
le z-score pour identifier les chiffres anormalement fréquents.

Implémentation en Python pur (pas SQL).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Distribution attendue selon la loi de Benford
BENFORD_EXPECTED = {
    1: 0.30103,
    2: 0.17609,
    3: 0.12494,
    4: 0.09691,
    5: 0.07918,
    6: 0.06695,
    7: 0.05799,
    8: 0.05115,
    9: 0.04576,
}


def _first_digit(x: float) -> int:
    """Retourne le premier chiffre significatif d'un nombre positif."""
    if x <= 0:
        return 0
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def _classify_mad(mad: float) -> str:
    """Classifie le MAD selon les seuils de Nigrini."""
    if mad < 0.006:
        return "conforme"
    elif mad < 0.012:
        return "acceptable"
    elif mad < 0.015:
        return "marginal"
    else:
        return "non_conforme"


def run_t1_benford(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Exécute le test de Benford sur les montants du grand livre.

    Args:
        df: Grand livre nettoyé.
        config: Section t1_benford du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    min_lines = config.get("min_lines_per_group", 200)
    z_threshold = config.get("z_score_threshold", 2.58)
    prefix_len = config.get("account_prefix_length", 3)
    min_deviation = config.get("min_deviation", 0.01)
    min_mad_status = config.get("min_mad_status", "non_conforme")

    results = []

    # Calculer les montants (prendre le max de debit/credit par ligne)
    df = df.copy()
    df["amount"] = df[["debit", "credit"]].max(axis=1)
    df = df[df["amount"] > 0].copy()

    # Extraire le premier chiffre
    df["first_digit"] = df["amount"].apply(_first_digit)
    df = df[df["first_digit"] >= 1]

    # Grouper par préfixe de compte (longueur configurable, par défaut 3)
    df["account_group"] = df["account"].astype(str).str[:prefix_len]

    for group, group_df in df.groupby("account_group"):
        n = len(group_df)
        if n < min_lines:
            logger.debug(
                "Groupe '%s' ignoré (%d lignes < %d minimum)", group, n, min_lines
            )
            continue

        # Distribution observée
        observed = group_df["first_digit"].value_counts(normalize=True).to_dict()

        # Calculer le MAD
        deviations = []
        digit_analysis = {}
        for digit in range(1, 10):
            obs = observed.get(digit, 0.0)
            exp = BENFORD_EXPECTED[digit]
            dev = abs(obs - exp)
            deviations.append(dev)

            # Z-score pour ce chiffre
            n_digit = group_df["first_digit"].eq(digit).sum()
            se = np.sqrt(exp * (1 - exp) / n)
            z_score = abs(obs - exp) / se if se > 0 else 0

            digit_analysis[digit] = {
                "observed": obs,
                "expected": exp,
                "deviation": dev,
                "z_score": z_score,
                "count": n_digit,
            }

        mad = np.mean(deviations)
        classification = _classify_mad(mad)

        eligible_statuses = (
            ("non_conforme",)
            if min_mad_status == "non_conforme"
            else ("marginal", "non_conforme")
        )

        if classification in eligible_statuses:
            # Signaler les chiffres avec un écart significatif
            for digit, analysis in digit_analysis.items():
                if (
                    analysis["z_score"] > z_threshold
                    and analysis["deviation"] > min_deviation
                ):
                    # Trouver les écritures avec ce chiffre dans ce groupe
                    flagged = group_df[group_df["first_digit"] == digit]
                    entry_ids = flagged["entry_id"].unique()

                    for eid in entry_ids:
                        results.append({
                            "entry_id": eid,
                            "test_id": "T1",
                            "observed_value": (
                                f"chiffre={digit}, "
                                f"fréq={analysis['observed']:.4f}, "
                                f"attendu={analysis['expected']:.4f}"
                            ),
                            "threshold": (
                                f"MAD={mad:.4f} ({classification}), "
                                f"z={analysis['z_score']:.2f}>{z_threshold}"
                            ),
                            "message": (
                                f"Loi de Benford : le chiffre {digit} apparaît à "
                                f"{analysis['observed']:.1%} au lieu de {analysis['expected']:.1%} "
                                f"pour le groupe de comptes '{group}' "
                                f"(MAD={mad:.4f}, classification={classification})"
                            ),
                        })

            logger.info(
                "T1 Benford — Groupe '%s' : MAD=%.4f (%s), %d écritures signalées",
                group, mad, classification,
                len(set(r["entry_id"] for r in results if f"'{group}'" in r["message"])),
            )

    result_df = pd.DataFrame(
    results,
    columns=[
        "entry_id",
        "test_id",
        "observed_value",
        "threshold",
        "message",
    ],
)
    if not result_df.empty:
        result_df = (
           result_df.groupby(["entry_id", "test_id"], as_index=False)
           .agg(
               observed_value=("observed_value", lambda s: " ; ".join(s)),
               threshold=("threshold", lambda s: " ; ".join(s)),
               message=("message", lambda s: " ; ".join(s)),
        )
    )

    logger.info(
        "T1 Benford : %d écritures signalées",
         result_df["entry_id"].nunique() if not result_df.empty else 0,
)
    return result_df

