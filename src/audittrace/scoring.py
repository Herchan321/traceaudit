"""
scoring.py — Score de risque par écriture (F3).

Combine un score basé sur les règles (somme pondérée des tests déclenchés)
et un score statistique (Isolation Forest) pour produire un score final
normalisé de 0 à 100 avec trois niveaux de risque.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def _load_weights(config_dir: Path) -> dict:
    """Charge la configuration de scoring."""
    path = config_dir / "weights.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rule_score(
    test_results: pd.DataFrame,
    weights: dict[str, int],
) -> pd.DataFrame:
    """
    Calcule le score basé sur les règles pour chaque écriture.

    Le score est la somme pondérée des tests déclenchés, normalisé
    par le total maximum possible.

    Args:
        test_results: Résultats consolidés de tous les tests.
        weights: Dictionnaire test_id → poids.

    Returns:
        DataFrame avec entry_id, rule_score, tests_triggered.
    """
    if test_results.empty:
        return pd.DataFrame(columns=["entry_id", "rule_score", "tests_triggered", "n_tests"])

    # Mapper les test_id vers les clés de poids
    test_weight_map = {
        "T1": weights.get("T1_benford", 8),
        "T2": weights.get("T2_duplicates", 10),
        "T3": weights.get("T3_round_amounts", 5),
        "T4": weights.get("T4_unusual_hours", 6),
        "T5": weights.get("T5_splitting", 9),
        "T6": weights.get("T6_imbalance", 10),
        "T7": weights.get("T7_rare_pairs", 7),
        "T8": weights.get("T8_late_posting", 6),
        "T9": weights.get("T9_atypical_user", 7),
    }

    max_total = sum(test_weight_map.values())

    # Agréger par écriture
    grouped = test_results.groupby("entry_id").agg(
        tests_triggered=("test_id", lambda x: ",".join(sorted(set(x)))),
        n_tests=("test_id", "nunique"),
    ).reset_index()

    # Calculer le score pondéré
    def _calc_score(tests_str: str) -> float:
        tests = tests_str.split(",")
        return sum(test_weight_map.get(t.strip(), 0) for t in tests)

    grouped["raw_score"] = grouped["tests_triggered"].apply(_calc_score)
    grouped["rule_score"] = (grouped["raw_score"] / max_total * 100).clip(0, 100)

    return grouped[["entry_id", "rule_score", "tests_triggered", "n_tests"]]


def compute_iforest_score(
    df: pd.DataFrame,
    contamination: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Calcule un score d'anomalie via Isolation Forest.

    Les features utilisées sont des variables dérivées du grand livre :
    montant, heure de saisie, jour de la semaine, écart document/comptabilisation,
    volume de l'utilisateur.

    Args:
        df: Grand livre nettoyé.
        contamination: Proportion estimée d'anomalies.
        seed: Graine aléatoire.

    Returns:
        DataFrame avec entry_id et iforest_score (0-100).
    """
    work = df.copy()
    work["amount"] = work[["debit", "credit"]].max(axis=1)
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work["posting_date"] = pd.to_datetime(work["posting_date"], errors="coerce")
    work["document_date"] = pd.to_datetime(work["document_date"], errors="coerce")

    # Features par écriture
    entry_features = work.groupby("entry_id").agg(
        max_amount=("amount", "max"),
        sum_amount=("amount", "sum"),
        n_lines=("line_id", "count"),
        hour=("created_at", lambda x: x.iloc[0].hour if pd.notna(x.iloc[0]) else 12),
        weekday=("created_at", lambda x: x.iloc[0].weekday() if pd.notna(x.iloc[0]) else 2),
        day_gap=("posting_date", lambda x: (
            (x.iloc[0] - work.loc[x.index[0], "document_date"]).days
            if pd.notna(x.iloc[0]) and pd.notna(work.loc[x.index[0], "document_date"])
            else 0
        )),
        user_id=("user_id", "first"),
    ).reset_index()

    # Volume par utilisateur
    user_volume = work.groupby("user_id")["entry_id"].nunique().reset_index()
    user_volume.columns = ["user_id", "user_entry_count"]
    entry_features = entry_features.merge(user_volume, on="user_id", how="left")

    # Montant moyen par utilisateur
    user_avg = work.groupby("user_id")["amount"].mean().reset_index()
    user_avg.columns = ["user_id", "user_avg_amount"]
    entry_features = entry_features.merge(user_avg, on="user_id", how="left")

    # Préparer la matrice de features
    feature_cols = [
        "max_amount", "sum_amount", "n_lines", "hour",
        "weekday", "day_gap", "user_entry_count", "user_avg_amount",
    ]

    X = entry_features[feature_cols].fillna(0).values

    # Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
        n_jobs=-1,
    )
    clf.fit(X)

    # Score d'anomalie : decision_function retourne des valeurs négatives pour anomalies
    raw_scores = -clf.decision_function(X)  # Inverser pour que + élevé = + anormal
    scaler = MinMaxScaler(feature_range=(0, 100))
    entry_features["iforest_score"] = scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()

    return entry_features[["entry_id", "iforest_score"]]


def compute_final_scores(
    df: pd.DataFrame,
    test_results: pd.DataFrame,
    config_dir: Path,
) -> pd.DataFrame:
    """
    Calcule le score de risque final combinant règles et Isolation Forest.

    Args:
        df: Grand livre nettoyé.
        test_results: Résultats consolidés de tous les tests.
        config_dir: Chemin vers le dossier de configuration.

    Returns:
        DataFrame avec entry_id, rule_score, iforest_score, final_score,
        risk_level, tests_triggered, n_tests.
    """
    config = _load_weights(config_dir)
    weights = config["test_weights"]
    scoring = config["scoring"]
    risk_levels = config["risk_levels"]

    logger.info("Calcul des scores de risque...")

    # Score règles
    rule_scores = compute_rule_score(test_results, weights)
    logger.info("Score règles calculé pour %d écritures", len(rule_scores))

    # Score Isolation Forest
    iforest_scores = compute_iforest_score(
        df,
        contamination=scoring.get("contamination", 0.02),
    )
    logger.info("Score Isolation Forest calculé pour %d écritures", len(iforest_scores))

    # Toutes les écritures uniques
    all_entries = pd.DataFrame({"entry_id": df["entry_id"].unique()})

    # Joindre les scores
    scores = all_entries.merge(rule_scores, on="entry_id", how="left")
    scores = scores.merge(iforest_scores, on="entry_id", how="left")

    # Remplir les valeurs manquantes
    scores["rule_score"] = scores["rule_score"].fillna(0)
    scores["iforest_score"] = scores["iforest_score"].fillna(0)
    scores["tests_triggered"] = scores["tests_triggered"].fillna("")
    scores["n_tests"] = scores["n_tests"].fillna(0).astype(int)

    # Score final combiné
    rule_weight = scoring.get("rule_weight", 0.6)
    iforest_weight = scoring.get("iforest_weight", 0.4)

    scores["final_score"] = (
        scores["rule_score"] * rule_weight
        + scores["iforest_score"] * iforest_weight
    ).clip(0, 100).round(1)

    # Niveaux de risque
    low_max = risk_levels.get("low_max", 30)
    medium_max = risk_levels.get("medium_max", 60)

    scores["risk_level"] = pd.cut(
        scores["final_score"],
        bins=[-1, low_max, medium_max, 101],
        labels=["faible", "moyen", "élevé"],
    )

    # Statistiques
    for level in ["faible", "moyen", "élevé"]:
        count = (scores["risk_level"] == level).sum()
        pct = count / len(scores) * 100
        logger.info("  %s : %d écritures (%.1f%%)", level, count, pct)

    return scores
