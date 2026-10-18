
"""Interface en ligne de commande d'AuditTrace."""

import logging
import sys
from pathlib import Path

# Assurer l'encodage UTF-8 sur terminal Windows pour les caractères français et typographiques
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
import pandas as pd
import yaml

from audittrace.agent.core import AuditAgent
from audittrace.agent.memo import generate_audit_memorandum
from audittrace.audit_tests.t1_benford import run_t1_benford
from audittrace.audit_tests.t2_duplicates import run_t2_duplicates
from audittrace.audit_tests.t3_round_amounts import run_t3_round_amounts
from audittrace.audit_tests.t4_unusual_hours import run_t4_unusual_hours
from audittrace.audit_tests.t5_splitting import run_t5_splitting
from audittrace.audit_tests.t6_imbalance import run_t6_imbalance
from audittrace.audit_tests.t7_rare_pairs import run_t7_rare_pairs
from audittrace.audit_tests.t8_late_posting import run_t8_late_posting
from audittrace.audit_tests.t9_atypical_user import run_t9_atypical_user
from audittrace.evaluate import run_evaluation
from audittrace.export import export_to_excel
from audittrace.generate_data import generate_synthetic_data
from audittrace.ingest import ingest_csv
from audittrace.quality import run_quality_checks
from audittrace.scoring import compute_final_scores

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Racine du projet, indépendamment du dossier courant.
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EXPECTED_COLUMNS = [
    "entry_id",
    "test_id",
    "observed_value",
    "threshold",
    "message",
]


@click.group()
@click.version_option(version="1.0.0", prog_name="audittrace")
def main():
    """AuditTrace : détection et traçabilité des anomalies comptables."""


@main.command()
def hello():
    """Vérifier que l'installation fonctionne."""
    click.echo("AuditTrace fonctionne correctement !")


@main.command()
def run():
    """Exécuter le pipeline complet sur des données synthétiques."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    click.echo("\n=== 1/6 - Génération des données synthétiques ===")
    ledger_path, ground_truth_path = generate_synthetic_data(
        CONFIG_DIR, OUTPUT_DIR
    )
    click.echo(f"Grand livre : {ledger_path}")
    click.echo(f"Référentiel d'évaluation : {ground_truth_path}")

    click.echo("\n=== 2/6 - Import et normalisation ===")
    df = ingest_csv(ledger_path, CONFIG_DIR)
    click.echo(f"{len(df):,} lignes importées".replace(",", " "))

    click.echo("\n=== 3/6 - Contrôles qualité ===")
    clean_df, quality_report = run_quality_checks(df, OUTPUT_DIR)
    click.echo(f"{len(clean_df):,} lignes conservées".replace(",", " "))
    click.echo(f"Problèmes qualité : {len(quality_report)}")

    click.echo("\n=== 4/6 - Exécution des 9 tests d'audit ===")
    with open(CONFIG_DIR / "thresholds.yaml", encoding="utf-8") as file:
        thresholds = yaml.safe_load(file) or {}

    tests = [
        ("t1_benford", run_t1_benford),
        ("t2_duplicates", run_t2_duplicates),
        ("t3_round_amounts", run_t3_round_amounts),
        ("t4_unusual_hours", run_t4_unusual_hours),
        ("t5_splitting", run_t5_splitting),
        ("t6_imbalance", run_t6_imbalance),
        ("t7_rare_pairs", run_t7_rare_pairs),
        ("t8_late_posting", run_t8_late_posting),
        ("t9_atypical_user", run_t9_atypical_user),
    ]

    results = []

    for test_name, test_function in tests:
        test_config = thresholds.get(test_name, {})
        result = test_function(clean_df, test_config)

        if result is not None and not result.empty:
            results.append(result)

        count = 0 if result is None else len(result)
        click.echo(f"{test_name}: {count} anomalie(s) signalée(s)")

    if results:
        test_results = pd.concat(results, ignore_index=True)
    else:
        test_results = pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Garantir un schéma stable, même si certains tests ne trouvent rien.
    for column in EXPECTED_COLUMNS:
        if column not in test_results.columns:
            test_results[column] = pd.NA

    test_results = test_results[EXPECTED_COLUMNS]

    test_results.to_csv(
        OUTPUT_DIR / "test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    click.echo(f"Total des signalements : {len(test_results)}")

    click.echo("\n=== 5/6 - Calcul des scores et évaluation ===")
    scores = compute_final_scores(clean_df, test_results, CONFIG_DIR)
    evaluation = run_evaluation(
        test_results, ground_truth_path, OUTPUT_DIR
    )

    click.echo(f"Écritures scorées : {len(scores)}")
    click.echo("Évaluation terminée.")

    click.echo("\n=== 6/6 - Export du rapport Excel ===")
    report_path = OUTPUT_DIR / "audittrace_report.xlsx"

    export_to_excel(
        scores,
        test_results,
        quality_report,
        evaluation,
        report_path,
    )

    click.echo("\nPipeline terminé !")
    click.echo(f"Résultats disponibles dans : {OUTPUT_DIR}")
    click.echo(f"Rapport Excel : {report_path}")


@main.command()
@click.argument("prompt", required=False)
def chat(prompt: str | None):
    """Copilote d'audit interactif : poser des questions sur les anomalies (ISA 240)."""
    agent = AuditAgent()
    click.echo("=== [AuditTrace Copilot] Agent d'Audit ISA 240 ===")
    click.echo("Posez vos questions d'investigation ou tapez 'exit' pour quitter.\n")

    if prompt:
        response = agent.run(prompt)
        click.echo(f"\n{response}\n")
        return

    while True:
        try:
            user_input = click.prompt("Auditeur", prompt_suffix=" > ")
            if user_input.strip().lower() in ("exit", "quit", "q"):
                click.echo("Fin de session.")
                break
            if not user_input.strip():
                continue
            response = agent.run(user_input)
            click.echo(f"\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            click.echo("\nFin de session.")
            break


@main.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Chemin du mémorandum")
def memo(output: Path | None):
    """Générer le Mémorandum d'Audit officiel conforme à la norme ISA 240."""
    click.echo("\n=== [AuditTrace] Génération du Mémorandum d'Audit ISA 240 ===")
    path = generate_audit_memorandum(output)
    click.echo(f"Mémorandum généré avec succès : {path}")


if __name__ == "__main__":
    main()
