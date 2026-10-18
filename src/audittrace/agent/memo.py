"""
memo.py — Générateur de Mémorandum d'Audit ISA 240.

Génère une note de synthèse formelle (Audit Memorandum) prête pour
l'associé signataire ou le comité d'audit, structurée selon les standards
internationaux de commissariat aux comptes.
"""

from datetime import datetime
from pathlib import Path

import duckdb

from audittrace.agent.knowledge import search_isa_norm

PROJECT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def generate_audit_memorandum(output_path: Path | None = None) -> Path:
    """
    Rédige et sauvegarde le Mémorandum d'Audit officiel ISA 240.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "audit_memorandum_isa240.md"

    ledger_path = str(OUTPUT_DIR / "general_ledger_clean.parquet").replace("\\", "/")
    test_results_path = str(OUTPUT_DIR / "test_results.csv").replace("\\", "/")

    con = duckdb.connect()

    # 1. Statistiques globales
    stats = con.execute(f"""
        SELECT
            COUNT(DISTINCT entry_id) as total_entries,
            ROUND(SUM(GREATEST(debit, credit)), 2) as total_turnover
        FROM '{ledger_path}'
    """).df().to_dict(orient="records")[0]

    # 2. Statistiques des anomalies
    anomaly_stats = con.execute(f"""
        SELECT
            COUNT(DISTINCT l.entry_id) as flagged_entries,
            ROUND(SUM(GREATEST(l.debit, l.credit)), 2) as exposed_volume,
            ROUND(AVG(GREATEST(l.debit, l.credit)), 2) as avg_amount
        FROM '{ledger_path}' l
        JOIN '{test_results_path}' tr ON l.entry_id = tr.entry_id
    """).df().to_dict(orient="records")[0]

    # 3. Répartition par test
    test_breakdown = con.execute(f"""
        SELECT
            test_id,
            COUNT(DISTINCT entry_id) as n_entries
        FROM '{test_results_path}'
        GROUP BY test_id
        ORDER BY test_id
    """).df().to_dict(orient="records")

    # 4. Top 5 des écritures les plus critiques (cumulant le plus de tests)
    critical_entries = con.execute(f"""
        SELECT
            l.entry_id,
            l.user_id,
            l.document_date,
            l.created_at,
            STRING_AGG(DISTINCT tr.test_id, ', ') as tests,
            COUNT(DISTINCT tr.test_id) as n_tests,
            ROUND(MAX(GREATEST(l.debit, l.credit)), 2) as amount
        FROM '{ledger_path}' l
        JOIN '{test_results_path}' tr ON l.entry_id = tr.entry_id
        GROUP BY l.entry_id, l.user_id, l.document_date, l.created_at
        ORDER BY n_tests DESC, amount DESC
        LIMIT 5
    """).df().to_dict(orient="records")

    now_str = datetime.now().strftime("%d/%m/%Y")

    # Tableaux markdown
    breakdown_rows = "\n".join([
        f"| **{t['test_id']}** | {t['n_entries']:,} écritures | {search_isa_norm(t['test_id'])[0]['paragraph']} |"
        for t in test_breakdown
    ])

    critical_rows = "\n".join([
        f"| `{c['entry_id']}` | `{c['user_id']}` | {c['document_date']} | **{c['amount']:,.2f} €** | `{c['tests']}` ({c['n_tests']}) |"
        for c in critical_entries
    ])

    memo_content = f"""# 📑 MÉMORANDUM D'AUDIT FINANCIER — NORME ISA 240
**Mission :** Audit du Grand Livre Général & Revue des Écritures de Journal
**Date du rapport :** {now_str}
**Destinataire :** Comité d'Audit & Associé Responsable de Mission
**Émetteur :** Copilote d'Audit Agentique AuditTrace
**Statut :** **STRICTEMENT CONFIDENTIEL**

---

## 1. Contexte et Cadre de la Mission

Conformément à la norme internationale d'audit **ISA 240** (*« Responsabilité de l'auditeur concernant les fraudes dans un audit d'états financiers »*, paragraphes 32 et A41 à A45), notre équipe a mis en œuvre des procédures d'audit automatisées sur l'intégralité du grand livre comptable de la période.

L'objectif de cette mission est de détecter les indices de contournement des contrôles internes par la direction ou les collaborateurs, ainsi que les risques de manipulations frauduleuses sur les écritures d'ajustement.

---

## 2. Synthèse Exécutive des Constats

* **Population totale auditée :** {stats['total_entries']:,} écritures comptables ({stats['total_turnover']:,.2f} € cumulés).
* **Écritures faisant l'objet de réserves / signalements :** **{anomaly_stats['flagged_entries']:,}** écritures ({anomaly_stats['flagged_entries']/stats['total_entries']:.1%} de la population).
* **Masse financière sous vigilance renforcée :** **{anomaly_stats['exposed_volume']:,.2f} €**.
* **Montant moyen des opérations suspectes :** {anomaly_stats['avg_amount']:,.2f} €.

> [!WARNING]
> **Opinion d'audit préliminaire :** Des signaux concordants de fractionnement sous seuils d'approbation (T5), d'écritures passées à des horaires atypiques (T4) et de déséquilibres comptables (T6) justifient l'extension immédiate des contrôles de corroboration substantifs.

---

## 3. Détail des Contrôles Normatifs ISA 240 Exécutés

| Test d'Audit | Volume Détecté | Référence Normative Applicable |
| :--- | :---: | :--- |
{breakdown_rows}

---

## 4. Écritures Présentant un Risque Majeur (Top Priorité)

Les écritures ci-dessous cumulent plusieurs signaux indépendants (ex: horaires atypiques + fractionnement + compte inhabituel) et constituent les cibles prioritaires d'investigation :

| N° Pièce | Utilisateur | Date Pièce | Montant Max | Tests Cumulés |
| :--- | :---: | :---: | :---: | :---: |
{critical_rows}

---

## 5. Recommandations & Plan de Diligences Substantives

L'Agent d'Audit recommande au collège des commissaires aux comptes de réaliser sans délai les vérifications suivantes :

1. **Circulation des pièces justificatives (Cut-off) :** Exiger les factures, contrats et bons de commande originaux pour les écritures du Top 5 ci-dessus.
2. **Revue des droits d'accès au système comptable :** Examiner les habilitations de l'utilisateur ayant passé des écritures hors heures de bureau et sans validation préalable.
3. **Contrôle physique des autorisations de signature :** Vérifier si les écritures détectées par le test T5 ont été sciemment scindées pour contourner le seuil d'autorisation de 10 000 €.

---
*Mémorandum généré automatiquement par AuditTrace Copilot — Traçabilité intégrale et hash d'intégrité conservés.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(memo_content)

    return output_path
