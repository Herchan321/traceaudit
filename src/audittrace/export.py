"""
export.py — Export des résultats en fichier Excel multi-onglets (F7).

Produit un classeur Excel avec un onglet par test, un onglet synthèse
et un onglet qualité des données, utilisant XlsxWriter pour le formatage.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def export_to_excel(
    scores: pd.DataFrame,
    test_results: pd.DataFrame,
    quality_report: pd.DataFrame,
    evaluation: dict,
    output_path: Path,
) -> Path:
    """
    Exporte les résultats complets dans un fichier Excel.

    Args:
        scores: Scores de risque par écriture.
        test_results: Résultats de tous les tests consolidés.
        quality_report: Rapport qualité des données.
        evaluation: Métriques d'évaluation.
        output_path: Chemin du fichier Excel de sortie.

    Returns:
        Chemin du fichier Excel créé.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(output_path), engine="xlsxwriter") as writer:
        workbook = writer.book

        # ── Formats ──
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#1B2A4A",
            "font_color": "#FFFFFF",
            "border": 1,
            "text_wrap": True,
            "valign": "vcenter",
        })
        risk_high = workbook.add_format({"bg_color": "#FF6B6B", "font_color": "#FFFFFF"})
        risk_medium = workbook.add_format({"bg_color": "#FFD93D"})
        risk_low = workbook.add_format({"bg_color": "#6BCB77", "font_color": "#FFFFFF"})
        number_fmt = workbook.add_format({"num_format": "#,##0.00"})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})

        # ── Onglet Synthèse ──
        _write_summary_sheet(
    writer, workbook, scores, test_results, evaluation, header_fmt, pct_fmt
)
        # ── Onglet par test ──
        for test_id in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]:
            test_data = test_results[test_results["test_id"] == test_id]
            if not test_data.empty:
                sheet_name = test_id

                test_data.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=False,
                    startrow=1,
                )
                ws = writer.sheets[sheet_name]

                for col_idx, col_name in enumerate(test_data.columns):
                    ws.write(0, col_idx, col_name, header_fmt)

                    max_len = max(
                        test_data[col_name].astype(str).str.len().max(),
                        len(col_name),
                    )
                    ws.set_column(col_idx, col_idx, min(max_len + 2, 60))

                ws.freeze_panes(1, 0)
                ws.autofilter(
                    0, 0, len(test_data), len(test_data.columns) - 1
                )

        # ── Onglet Scores ──
        _write_scores_sheet(
            writer, scores, header_fmt, risk_high, risk_medium, risk_low, number_fmt
        )

        # ── Onglet Qualité ──
        if not quality_report.empty:
            quality_report.to_excel(writer, sheet_name="Qualité", index=False, startrow=1)
            ws = writer.sheets["Qualité"]
            for col_idx, col_name in enumerate(quality_report.columns):
                ws.write(0, col_idx, col_name, header_fmt)

    return output_path


def _write_summary_sheet(
    writer, workbook, scores, test_results, evaluation, header_fmt, pct_fmt
):
    """Écrit l'onglet de synthèse."""
    ws = workbook.add_worksheet("Synthèse")
    writer.sheets["Synthèse"] = ws

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 16, "font_color": "#1B2A4A",
    })
    subtitle_fmt = workbook.add_format({
        "bold": True, "font_size": 12, "font_color": "#1B2A4A",
        "bottom": 2, "bottom_color": "#1B2A4A",
    })
    bold = workbook.add_format({"bold": True})

    row = 0
    ws.write(row, 0, "AuditTrace — Rapport de synthèse", title_fmt)
    row += 2

    # Statistiques générales
    ws.write(row, 0, "Statistiques générales", subtitle_fmt)
    ws.merge_range(row, 0, row, 3, "Statistiques générales", subtitle_fmt)
    row += 1

    n_total = len(scores)
    n_high = (scores["risk_level"] == "élevé").sum() if "risk_level" in scores.columns else 0
    n_medium = (scores["risk_level"] == "moyen").sum() if "risk_level" in scores.columns else 0
    n_low = (scores["risk_level"] == "faible").sum() if "risk_level" in scores.columns else 0
    n_flagged = (
        test_results["entry_id"].nunique()
        if not test_results.empty and "entry_id" in test_results.columns
        else 0
    )
    n_alerts = len(test_results)

    stats = [
        ("Nombre total d'écritures", n_total),
        ("Risque élevé", f"{n_high} ({n_high / n_total * 100:.1f}%)" if n_total > 0 else "0"),
        ("Risque moyen", f"{n_medium} ({n_medium / n_total * 100:.1f}%)" if n_total > 0 else "0"),
        ("Risque faible", f"{n_low} ({n_low / n_total * 100:.1f}%)" if n_total > 0 else "0"),
        ("Écritures avec au moins une alerte", n_flagged),
        ("Nombre total de signalements", n_alerts),
    ]
    for label, value in stats:
        ws.write(row, 0, label, bold)
        ws.write(row, 1, str(value))
        row += 1

    row += 1

    # Métriques d'évaluation
    if evaluation and "per_test" in evaluation and evaluation["per_test"]:
        ws.write(row, 0, "Métriques d'évaluation par test", subtitle_fmt)
        ws.merge_range(row, 0, row, 5, "Métriques d'évaluation par test", subtitle_fmt)
        row += 1

        eval_headers = [
    "Test", "Précision", "Rappel", "F1",
    "Signalés", "Réels", "Vrais positifs",
    "Faux positifs", "Faux négatifs",
]
        for col_idx, h in enumerate(eval_headers):
            ws.write(row, col_idx, h, header_fmt)
        row += 1

        for t in evaluation["per_test"]:
            ws.write(row, 0, t["test_id"])
            ws.write(row, 1, t["precision"], pct_fmt)
            ws.write(row, 2, t["recall"], pct_fmt)
            ws.write(row, 3, t["f1_score"], pct_fmt)
            ws.write(row, 4, t["flagged_entries"])
            ws.write(row, 5, t["actual_anomalies"])
            ws.write(row, 6, t["true_positives"])
            ws.write(row, 7, t["false_positives"])
            ws.write(row, 8, t["false_negatives"])
            row += 1

    ws.set_column(0, 0, 35)
    ws.set_column(1, 8, 15)



def _write_scores_sheet(
    writer, scores, header_fmt, risk_high, risk_medium, risk_low, number_fmt
):
    """Écrit les scores avec filtres et mise en forme du risque."""
    if scores.empty:
        pd.DataFrame({"Information": ["Aucun score disponible"]}).to_excel(
            writer, sheet_name="Scores", index=False
        )
        return

    sorted_scores = scores.sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)

    # Écrire les données sans en-tête : nous créons l'en-tête nous-mêmes.
    sorted_scores.to_excel(
        writer,
        sheet_name="Scores",
        index=False,
        header=False,
        startrow=1,
    )

    ws = writer.sheets["Scores"]

    for col_idx, col_name in enumerate(sorted_scores.columns):
        ws.write(0, col_idx, col_name, header_fmt)

    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(sorted_scores), len(sorted_scores.columns) - 1)

    # Largeurs lisibles
    for col_idx, col_name in enumerate(sorted_scores.columns):
        if col_name == "entry_id":
            width = 18
        elif col_name == "tests_triggered":
            width = 40
        elif col_name == "risk_level":
            width = 14
        else:
            width = 16

        ws.set_column(col_idx, col_idx, width)

    # Mise en forme conditionnelle du niveau de risque
    if "risk_level" in sorted_scores.columns:
        risk_col = sorted_scores.columns.get_loc("risk_level")

        for row_idx, level in enumerate(sorted_scores["risk_level"], start=1):
            if level == "élevé":
                ws.write(row_idx, risk_col, level, risk_high)
            elif level == "moyen":
                ws.write(row_idx, risk_col, level, risk_medium)
            elif level == "faible":
                ws.write(row_idx, risk_col, level, risk_low)
