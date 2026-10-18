
"""
evaluate.py — Évaluation des performances de détection.

Compare les écritures signalées par les tests avec la vérité terrain.
Calcule précision, rappel et F1 au niveau des écritures uniques.

Important :
- L'évaluation utilise des données synthétiques annotées.
- Aucune séparation entraînement/test 50/50 n'est effectuée ici.
- Les résultats mesurent la détection sur le jeu de données évalué.
"""


import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les identifiants pour éviter les différences de format."""
    df = df.copy()

    for column in ["entry_id", "test_id", "target_test", "anomaly_type"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df



def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Charge le fichier de vérité terrain."""
    if not gt_path.exists():
        logger.warning("Fichier de vérité terrain introuvable : %s", gt_path)
        return pd.DataFrame()

    gt = pd.read_csv(gt_path, dtype=str)
    gt = normalize_identifiers(gt)
    logger.info("Vérité terrain : %d anomalies, %d écritures uniques",
                len(gt), gt["entry_id"].nunique())
    return gt


def evaluate_test(
    test_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    test_id: str,
) -> dict:
    """
    Évalue un test spécifique contre la vérité terrain.

    Args:
        test_results: Résultats du test pour un test_id donné.
        ground_truth: Vérité terrain filtrée sur le target_test correspondant.
        test_id: Identifiant du test (T1, T2, etc.).

    Returns:
        Dictionnaire avec les métriques.
    """
    test_entries = set(test_results[test_results["test_id"] == test_id]["entry_id"].unique())
    gt_entries = set(ground_truth[ground_truth["target_test"] == test_id]["entry_id"].unique())

    tp = len(test_entries & gt_entries)
    fp = len(test_entries - gt_entries)
    fn = len(gt_entries - test_entries)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "test_id": test_id,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "flagged_entries": len(test_entries),
        "actual_anomalies": len(gt_entries),
    }


def build_confusion_matrix(
    test_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit la matrice des tests déclenchés par type d'anomalie.

    Lignes : types d'anomalies (ground truth)
    Colonnes : tests déclenchés
    Valeurs : nombre d'écritures
    """
    if ground_truth.empty or test_results.empty:
        return pd.DataFrame()

    anomaly_types = ground_truth["anomaly_type"].unique()
    test_ids = sorted(test_results["test_id"].unique())

    matrix_data = []
    for atype in anomaly_types:
        gt_entries = set(ground_truth[ground_truth["anomaly_type"] == atype]["entry_id"].unique())
        row = {"anomaly_type": atype, "total_anomalies": len(gt_entries)}

        for tid in test_ids:
            flagged = set(test_results[test_results["test_id"] == tid]["entry_id"].unique())
            row[tid] = len(gt_entries & flagged)

        matrix_data.append(row)

    return pd.DataFrame(matrix_data)


def run_evaluation(
    test_results: pd.DataFrame,
    gt_path: Path,
    output_dir: Path,
) -> dict:
    """
    Exécute l'évaluation complète.

    Args:
        test_results: Résultats consolidés de tous les tests.
        gt_path: Chemin vers le fichier de vérité terrain.
        output_dir: Dossier de sortie pour les rapports.

    Returns:
        Dictionnaire avec les métriques globales et par test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(gt_path)
    test_results = normalize_identifiers(test_results)

    if ground_truth.empty:
        logger.warning("Pas de vérité terrain — évaluation impossible")
        return {"global": {}, "per_test": [], "matrix": pd.DataFrame()}

    # Évaluation par test
    per_test = []
    for test_id in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]:
        metrics = evaluate_test(test_results, ground_truth, test_id)
        per_test.append(metrics)
        logger.info(
            "  %s : P=%.2f, R=%.2f, F1=%.2f (%d signalés, %d réels)",
            test_id, metrics["precision"], metrics["recall"], metrics["f1_score"],
            metrics["flagged_entries"], metrics["actual_anomalies"],
        )

    # Évaluation globale
    all_flagged = set(test_results["entry_id"].unique())
    all_gt = set(ground_truth["entry_id"].unique())
    tp = len(all_flagged & all_gt)
    fp = len(all_flagged - all_gt)
    fn = len(all_gt - all_flagged)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    global_metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "total_flagged": len(all_flagged),
        "total_anomalies": len(all_gt),
    }

    logger.info(
        "Global : P=%.2f, R=%.2f, F1=%.2f (%d signalés, %d réels, %d TP, %d FP, %d FN)",
        precision, recall, f1, len(all_flagged), len(all_gt), tp, fp, fn,
    )

    # Matrice de confusion
    matrix = build_confusion_matrix(test_results, ground_truth)

    # Sauvegarder les résultats
    per_test_df = pd.DataFrame(per_test)
    per_test_df.to_csv(output_dir / "evaluation_per_test.csv", index=False)

    if not matrix.empty:
        matrix.to_csv(output_dir / "confusion_matrix.csv", index=False)

    # Générer le rapport markdown
    _generate_evaluation_report(global_metrics, per_test, matrix, output_dir)

    return {
        "global": global_metrics,
        "per_test": per_test,
        "matrix": matrix,
    }


def _generate_evaluation_report(
    global_metrics: dict,
    per_test: list[dict],
    matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Génère le rapport d'évaluation en markdown."""
    lines = [
        "# Rapport d'évaluation — AuditTrace",
        "",
        "## Métriques globales",
        "",
        "| Métrique | Valeur |",
        "|----------|--------|",
        f"| Précision | {global_metrics['precision']:.2%} |",
        f"| Rappel | {global_metrics['recall']:.2%} |",
        f"| F1-Score | {global_metrics['f1_score']:.2%} |",
        f"| Écritures signalées | {global_metrics['total_flagged']} |",
        f"| Anomalies réelles | {global_metrics['total_anomalies']} |",
        f"| Vrais positifs | {global_metrics['true_positives']} |",
        f"| Faux positifs | {global_metrics['false_positives']} |",
        f"| Faux négatifs | {global_metrics['false_negatives']} |",
        "",
        "## Métriques par test",
        "",
        "| Test | Précision | Rappel | F1 | Signalés | Réels |",
        "|------|-----------|--------|----|----------|-------|",
    ]

    for t in per_test:
        lines.append(
            f"| {t['test_id']} | {t['precision']:.2%} | {t['recall']:.2%} | "
            f"{t['f1_score']:.2%} | {t['flagged_entries']} | {t['actual_anomalies']} |"
        )

    if not matrix.empty:
        lines.extend([
            "",
            "## Matrice tests × anomalies",
            "",
            matrix.to_markdown(index=False),
        ])

    lines.extend([
        "",
        "## Limites et remarques",
        "",
        "- Les données sont **synthétiques** : les résultats ne sont pas directement",
        "  transposables à des données réelles.",
        "- Les anomalies injectées sont relativement faciles à détecter par construction.",
        "- Le score Isolation Forest est entraîné sur l'ensemble complet, ce qui peut",
        "  favoriser la détection (risque de surapprentissage limité par la combinaison",
        "  avec le score règles).",
        "- Certains faux positifs sont attendus : une écriture peut être signalée par",
        "  un test sans être une anomalie injectée, mais rester légitime à examiner.",
    ])

    report_path = output_dir / "evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Rapport d'évaluation : %s", report_path)
