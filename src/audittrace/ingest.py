"""
ingest.py — Import et normalisation du grand livre comptable.

Lit un fichier CSV, applique le mapping de colonnes, force les types,
gère l'encodage et prépare les données pour les contrôles qualité.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "entry_id", "line_id", "account", "account_label",
    "debit", "credit", "posting_date", "document_date",
    "created_at", "user_id", "source", "description",
]

DTYPES = {
    "entry_id": "string",
    "line_id": "int64",
    "account": "string",
    "account_label": "string",
    "user_id": "string",
    "source": "string",
    "description": "string",
}


def load_column_mapping(config_dir: Path) -> dict[str, str]:
    """Charge le fichier de mapping de colonnes."""
    mapping_path = config_dir / "columns_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        logger.info("Mapping de colonnes chargé depuis %s", mapping_path)
        return mapping
    logger.warning("Pas de fichier de mapping trouvé, utilisation des noms par défaut")
    return {col: col for col in REQUIRED_COLUMNS}


def ingest_csv(
    csv_path: Path,
    config_dir: Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Lit et normalise un fichier CSV de grand livre comptable.

    Args:
        csv_path: Chemin vers le fichier CSV.
        config_dir: Chemin vers le dossier de configuration.
        encoding: Encodage du fichier CSV.

    Returns:
        DataFrame avec colonnes normalisées et types forcés.

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
        ValueError: Si des colonnes obligatoires sont manquantes.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    logger.info("Lecture du fichier %s (encodage: %s)...", csv_path, encoding)

    # Essayer plusieurs encodages si le premier échoue
    encodings_to_try = [encoding, "utf-8-sig", "latin-1", "cp1252"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            logger.info("Fichier lu avec l'encodage %s", enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if df is None:
        raise ValueError(f"Impossible de lire le fichier avec les encodages : {encodings_to_try}")

    # Appliquer le mapping de colonnes
    mapping = load_column_mapping(config_dir)
    reverse_mapping = {v: k for k, v in mapping.items()}
    df = df.rename(columns=reverse_mapping)

    # Vérifier les colonnes obligatoires
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing}")

    # Forcer les types
    for col, dtype in DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors="ignore")

    # Types numériques
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0)

    # Types dates
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df["document_date"] = pd.to_datetime(df["document_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    logger.info(
        "Fichier ingéré : %d lignes, %d colonnes, %d écritures distinctes",
        len(df), len(df.columns), df["entry_id"].nunique(),
    )

    return df
