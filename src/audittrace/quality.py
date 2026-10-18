"""
quality.py — Contrôles qualité du grand livre comptable.

Vérifie la cohérence des données importées et produit un rapport qualité.
Les erreurs non bloquantes sont journalisées ; l'exécution continue.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les valeurs manquantes dans les colonnes obligatoires."""
    issues = []
    critical_cols = ["entry_id", "account", "posting_date", "debit", "credit"]

    for col in critical_cols:
        nulls = df[col].isna()
        if nulls.any():
            for idx in df[nulls].index:
                issues.append({
                    "check": "missing_value",
                    "severity": "ERROR",
                    "row": idx,
                    "column": col,
                    "entry_id": (
                        df.at[idx, "entry_id"]
                        if pd.notna(df.at[idx, "entry_id"])
                        else "N/A"
                    ),
                    "message": f"Valeur manquante dans la colonne '{col}'",
                })

    return pd.DataFrame(issues)


def check_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les dates invalides ou incohérentes."""
    issues = []

    for col in ["posting_date", "document_date", "created_at"]:
        original_nulls = df[col].isna()
        if original_nulls.any():
            for idx in df[original_nulls].index:
                issues.append({
                    "check": "invalid_date",
                    "severity": "WARNING",
                    "row": idx,
                    "column": col,
                    "entry_id": str(df.at[idx, "entry_id"]),
                    "message": f"Date invalide ou non parsable dans '{col}'",
                })

    # Vérifier que document_date <= posting_date (quand les deux existent)
    valid = df["posting_date"].notna() & df["document_date"].notna()
    future = df[valid & (df["document_date"] > df["posting_date"])]
    for idx in future.index:
        issues.append({
            "check": "date_coherence",
            "severity": "WARNING",
            "row": idx,
            "column": "document_date",
            "entry_id": str(df.at[idx, "entry_id"]),
            "message": (
                f"Date document ({df.at[idx, 'document_date'].date()}) "
                f"postérieure à date comptabilisation ({df.at[idx, 'posting_date'].date()})"
            ),
        })

    return pd.DataFrame(issues)


def check_negative_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les montants négatifs."""
    issues = []

    for col in ["debit", "credit"]:
        neg = df[df[col] < 0]
        for idx in neg.index:
            issues.append({
                "check": "negative_amount",
                "severity": "ERROR",
                "row": idx,
                "column": col,
                "entry_id": str(df.at[idx, "entry_id"]),
                "message": f"Montant négatif : {df.at[idx, col]}€ dans '{col}'",
            })

    return pd.DataFrame(issues)


def check_both_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les lignes avec débit ET crédit renseignés simultanément."""
    issues = []
    both = df[(df["debit"] > 0) & (df["credit"] > 0)]
    for idx in both.index:
        issues.append({
            "check": "both_debit_credit",
            "severity": "WARNING",
            "row": idx,
            "column": "debit/credit",
            "entry_id": str(df.at[idx, "entry_id"]),
            "message": (
                f"Débit ({df.at[idx, 'debit']}€) et crédit ({df.at[idx, 'credit']}€) "
                "renseignés simultanément"
            ),
        })

    return pd.DataFrame(issues)


def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les lignes strictement identiques (doublons de données)."""
    issues = []
    cols = ["entry_id", "line_id", "account", "debit", "credit", "posting_date"]
    existing_cols = [c for c in cols if c in df.columns]

    duplicates = df[df.duplicated(subset=existing_cols, keep=False)]
    if not duplicates.empty:
        for idx in duplicates.index:
            issues.append({
                "check": "duplicate_row",
                "severity": "WARNING",
                "row": idx,
                "column": "all",
                "entry_id": str(df.at[idx, "entry_id"]),
                "message": "Ligne potentiellement dupliquée",
            })

    return pd.DataFrame(issues)


def run_quality_checks(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute tous les contrôles qualité et produit le rapport.

    Args:
        df: Grand livre importé.
        output_dir: Dossier de sortie pour le rapport et le fichier nettoyé.

    Returns:
        Tuple (DataFrame nettoyé, DataFrame du rapport qualité).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exécution des contrôles qualité sur %d lignes...", len(df))

    # Exécuter chaque contrôle
    all_issues = pd.concat([
        check_missing_values(df),
        check_invalid_dates(df),
        check_negative_amounts(df),
        check_both_debit_credit(df),
        check_duplicate_rows(df),
    ], ignore_index=True)

    # Statistiques
    if not all_issues.empty:
        n_errors = len(all_issues[all_issues["severity"] == "ERROR"])
        n_warnings = len(all_issues[all_issues["severity"] == "WARNING"])
        logger.info("Contrôles qualité : %d erreurs, %d avertissements", n_errors, n_warnings)

        # Sauvegarder le rapport
        report_path = output_dir / "quality_report.csv"
        all_issues.to_csv(report_path, index=False, encoding="utf-8")
        logger.info("Rapport qualité sauvegardé : %s", report_path)
    else:
        logger.info("Aucun problème de qualité détecté ✓")
        all_issues = pd.DataFrame(
            columns=["check", "severity", "row", "column", "entry_id", "message"]
        )

    # Nettoyage : supprimer les lignes avec erreurs bloquantes
    clean_df = df.copy()
    if not all_issues.empty:
        error_rows = all_issues[all_issues["severity"] == "ERROR"]["row"].unique()
        if len(error_rows) > 0:
            clean_df = clean_df.drop(index=error_rows, errors="ignore")
            logger.info("Lignes supprimées pour erreurs bloquantes : %d", len(error_rows))

    # Sauvegarder le fichier nettoyé en Parquet
    parquet_path = output_dir / "general_ledger_clean.parquet"
    clean_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Grand livre nettoyé : %s (%d lignes)", parquet_path, len(clean_df))

    return clean_df, all_issues
