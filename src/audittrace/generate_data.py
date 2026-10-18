"""
generate_data.py — Générateur de données synthétiques pour AuditTrace.

Produit un grand livre comptable réaliste (~100 000 lignes) avec injection
contrôlée d'anomalies. Le fichier de vérité terrain (ground_truth.csv) est
généré séparément et n'est jamais lu par les tests d'audit.
"""

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Plan comptable simplifié (40 comptes)
# ──────────────────────────────────────────────────────────────────────
CHART_OF_ACCOUNTS: dict[str, str] = {
    "101000": "Capital social",
    "106000": "Réserves",
    "108000": "Compte de l'exploitant",
    "120000": "Résultat de l'exercice",
    "164000": "Emprunts bancaires",
    "201000": "Frais d'établissement",
    "211000": "Terrains",
    "213000": "Constructions",
    "215000": "Matériel et outillage",
    "218000": "Matériel de transport",
    "274000": "Prêts",
    "370000": "Stocks de marchandises",
    "401000": "Fournisseurs",
    "401100": "Fournisseurs — effets à payer",
    "404000": "Fournisseurs d'immobilisations",
    "411000": "Clients",
    "411100": "Clients — effets à recevoir",
    "421000": "Personnel — rémunérations dues",
    "431000": "Sécurité sociale",
    "437000": "Autres organismes sociaux",
    "441000": "État — impôts sur les bénéfices",
    "445600": "TVA déductible",
    "445700": "TVA collectée",
    "471000": "Comptes d'attente",
    "512000": "Banque",
    "512100": "Banque secondaire",
    "530000": "Caisse",
    "601000": "Achats de matières premières",
    "602000": "Achats de fournitures",
    "606000": "Achats non stockés",
    "613000": "Locations",
    "615000": "Entretien et réparations",
    "616000": "Primes d'assurance",
    "622000": "Honoraires",
    "625000": "Déplacements et missions",
    "641000": "Rémunérations du personnel",
    "645000": "Charges de sécurité sociale",
    "681000": "Dotations aux amortissements",
    "701000": "Ventes de produits finis",
    "707000": "Ventes de marchandises",
}

ACCOUNTS = list(CHART_OF_ACCOUNTS.keys())

# Paires de comptes courantes (débit, crédit)
COMMON_PAIRS: list[tuple[str, str]] = [
    ("601000", "401000"),  # Achat matières → Fournisseurs
    ("602000", "401000"),  # Achat fournitures → Fournisseurs
    ("606000", "401000"),  # Achats non stockés → Fournisseurs
    ("401000", "512000"),  # Paiement fournisseur
    ("411000", "701000"),  # Vente produits → Clients
    ("411000", "707000"),  # Vente marchandises → Clients
    ("512000", "411000"),  # Encaissement client
    ("641000", "421000"),  # Salaires → Personnel
    ("645000", "431000"),  # Charges sociales → Sécu
    ("421000", "512000"),  # Paiement salaires
    ("431000", "512000"),  # Paiement sécu
    ("625000", "512000"),  # Déplacements → Banque
    ("613000", "512000"),  # Loyer → Banque
    ("615000", "512000"),  # Entretien → Banque
    ("616000", "512000"),  # Assurance → Banque
    ("622000", "401000"),  # Honoraires → Fournisseurs
    ("681000", "213000"),  # Amortissement constructions
    ("681000", "215000"),  # Amortissement matériel
    ("445600", "401000"),  # TVA déductible
    ("411000", "445700"),  # TVA collectée
    ("512000", "530000"),  # Caisse → Banque
    ("530000", "512000"),  # Banque → Caisse
    ("441000", "512000"),  # Impôts → Banque
    ("370000", "601000"),  # Stock → Achats
]

USERS = [f"user_{i:02d}" for i in range(1, 16)]


def _load_thresholds(config_dir: Path) -> dict:
    """Charge les seuils depuis le fichier de configuration."""
    path = config_dir / "thresholds.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _random_business_datetime(
    rng: np.random.Generator,
    date: datetime,
) -> datetime:
    """Génère un horodatage de saisie aléatoire en heures ouvrées."""
    hour = rng.integers(8, 19)
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return date.replace(hour=int(hour), minute=int(minute), second=int(second))


def _next_weekday(date: datetime) -> datetime:
    """Décale une date au prochain jour ouvré si elle tombe le week-end."""
    while date.weekday() >= 5:
        date += timedelta(days=1)
    return date


def _generate_amount(rng: np.random.Generator) -> float:
    """Génère un montant réaliste suivant une distribution log-normale."""
    # La distribution log-normale produit naturellement une distribution
    # des premiers chiffres proche de la loi de Benford
    amount = float(np.exp(rng.normal(6.5, 1.8)))
    return round(max(0.01, amount), 2)


def generate_base_entries(
    rng: np.random.Generator,
    n_entries: int,
    start_date: datetime,
    end_date: datetime,
    holidays: set[str] | None = None,
) -> pd.DataFrame:
    """Génère les écritures comptables de base (normales)."""
    rows: list[dict] = []
    entry_counter = 0
    days_range = (end_date - start_date).days

    for _ in range(n_entries):
        entry_counter += 1
        entry_id = f"JE-{entry_counter:06d}"

        # Date du document
        doc_offset = rng.integers(0, days_range)
        document_date = start_date + timedelta(days=int(doc_offset))

        # Date de comptabilisation (0-5 jours après le document)
        posting_offset = int(rng.integers(0, 6))
        posting_date = document_date + timedelta(days=posting_offset)

        # Saisie le même jour ou le jour suivant la comptabilisation, heures ouvrées
        created_date = posting_date + timedelta(days=int(rng.integers(0, 2)))
        # S'assurer que ce n'est pas un week-end ni un jour férié pour les écritures normales
        while (
            created_date.weekday() >= 5
            or (holidays and created_date.strftime("%m-%d") in holidays)
        ):
            created_date += timedelta(days=1)
        created_at = _random_business_datetime(rng, created_date)

        user = rng.choice(USERS)
        source = rng.choice(["automatique", "manuelle"], p=[0.7, 0.3])

        # Choisir une paire de comptes
        pair_idx = int(rng.integers(0, len(COMMON_PAIRS)))
        debit_account, credit_account = COMMON_PAIRS[pair_idx]

        amount = _generate_amount(rng)

        # Nombre de lignes (2 pour simple, parfois 3-4 pour écritures composées)
        n_lines = int(rng.choice([2, 2, 2, 2, 3, 4], p=[0.6, 0.15, 0.1, 0.05, 0.07, 0.03]))

        if n_lines == 2:
            # Écriture simple : 1 débit, 1 crédit
            rows.append({
                "entry_id": entry_id,
                "line_id": 1,
                "account": debit_account,
                "account_label": CHART_OF_ACCOUNTS[debit_account],
                "debit": amount,
                "credit": 0.0,
                "posting_date": posting_date.strftime("%Y-%m-%d"),
                "document_date": document_date.strftime("%Y-%m-%d"),
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user,
                "source": source,
                "description": f"Écriture {entry_id}",
            })
            rows.append({
                "entry_id": entry_id,
                "line_id": 2,
                "account": credit_account,
                "account_label": CHART_OF_ACCOUNTS[credit_account],
                "debit": 0.0,
                "credit": amount,
                "posting_date": posting_date.strftime("%Y-%m-%d"),
                "document_date": document_date.strftime("%Y-%m-%d"),
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user,
                "source": source,
                "description": f"Écriture {entry_id}",
            })
        else:
            # Écriture composée : répartir le montant entre les débits
            # et les crédits en conservant l'équilibre comptable.
            mid = n_lines // 2
            n_debits = mid
            n_credits = n_lines - mid

            def split_amount(total: float, count: int) -> list[float]:
                """Répartit un montant positif en parts positives équilibrées."""
                if count == 1:
                    return [round(float(total), 2)]

                # Générer des parts aléatoires strictement positives.
                weights = rng.uniform(0.5, 1.5, size=count)
                weights = weights / weights.sum()

                parts = [
                    round(float(total) * float(weights[i]), 2)
                    for i in range(count - 1)
                ]

                # La dernière part corrige l'arrondi au centime.
                last_part = round(float(total) - sum(parts), 2)
                parts.append(last_part)

                # Sécurité : éviter toute part nulle ou négative.
                if any(part <= 0 for part in parts):
                    parts = [round(float(total) / count, 2)] * count
                    parts[-1] = round(float(total) - sum(parts[:-1]), 2)

                return parts

            debit_amounts = split_amount(amount, n_debits)
            credit_amounts_adj = split_amount(amount, n_credits)

            available_accounts = [a for a in ACCOUNTS if a != debit_account and a != credit_account]
            extra_accounts = list(
                rng.choice(available_accounts, size=max(0, n_lines - 2), replace=False)
            )

            line_id = 0
            for i in range(mid):
                line_id += 1
                acc = (
                    debit_account
                    if i == 0
                    else (extra_accounts.pop() if extra_accounts else debit_account)
                )
                rows.append({
                    "entry_id": entry_id,
                    "line_id": line_id,
                    "account": acc,
                    "account_label": CHART_OF_ACCOUNTS.get(acc, "Compte divers"),
                    "debit": debit_amounts[i],
                    "credit": 0.0,
                    "posting_date": posting_date.strftime("%Y-%m-%d"),
                    "document_date": document_date.strftime("%Y-%m-%d"),
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": user,
                    "source": source,
                    "description": f"Écriture {entry_id}",
                })
            for i in range(n_lines - mid):
                line_id += 1
                acc = (
                    credit_account
                    if i == 0
                    else (extra_accounts.pop() if extra_accounts else credit_account)
                )
                rows.append({
                    "entry_id": entry_id,
                    "line_id": line_id,
                    "account": acc,
                    "account_label": CHART_OF_ACCOUNTS.get(acc, "Compte divers"),
                    "debit": 0.0,
                    "credit": credit_amounts_adj[i] if i < len(credit_amounts_adj) else 0.0,
                    "posting_date": posting_date.strftime("%Y-%m-%d"),
                    "document_date": document_date.strftime("%Y-%m-%d"),
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": user,
                    "source": source,
                    "description": f"Écriture {entry_id}",
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Injection d'anomalies
# ──────────────────────────────────────────────────────────────────────

def _inject_duplicates(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T2 : Injecte des écritures dupliquées."""
    candidates = df["entry_id"].unique()
    chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    new_rows = []

    for eid in chosen:
        original = df[df["entry_id"] == eid].copy()
        new_eid = f"{eid}-DUP"
        original["entry_id"] = new_eid
        new_rows.append(original)
        for _, row in original.iterrows():
            ground_truth.append({
                "entry_id": new_eid,
                "line_id": row["line_id"],
                "anomaly_type": "duplicate",
                "target_test": "T2",
                "description": f"Doublon de {eid}",
            })

    if new_rows:
        return pd.concat([df] + new_rows, ignore_index=True)
    return df


def _inject_round_amounts(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
    sensitive_prefixes: list[str],
) -> pd.DataFrame:
    """T3 : Injecte des montants ronds en conservant l'équilibre comptable."""
    mask = df["account"].apply(
        lambda a: any(str(a).startswith(p) for p in sensitive_prefixes)
    )
    candidates = df[mask].index.to_numpy()
    chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)

    for idx in chosen:
        row = df.loc[idx]
        eid = row["entry_id"]

        # On limite ce scénario aux écritures simples à deux lignes.
        entry = df[df["entry_id"] == eid]
        if len(entry) != 2:
            continue

        round_amount = float(
            rng.choice([1000, 2000, 3000, 5000, 10000, 15000, 20000, 50000])
        )

        other_idx = entry.index[entry.index != idx][0]

        if row["debit"] > 0:
            df.at[idx, "debit"] = round_amount
            df.at[other_idx, "credit"] = round_amount
        elif row["credit"] > 0:
            df.at[idx, "credit"] = round_amount
            df.at[other_idx, "debit"] = round_amount
        else:
            continue

        ground_truth.append({
            "entry_id": eid,
            "line_id": row["line_id"],
            "anomaly_type": "round_amount",
            "target_test": "T3",
            "description": f"Montant rond {round_amount}€ sur compte {row['account']}",
        })

    return df



def _inject_unusual_hours(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
    holidays: list[str] | None = None,
) -> pd.DataFrame:
    """T4 : Injecte des saisies nocturnes, de week-end ou de jours fériés."""
    if n <= 0 or df.empty:
        return df

    # Choisir des écritures distinctes pour éviter d'injecter
    # plusieurs fois la même écriture.
    entry_ids = df["entry_id"].drop_duplicates().to_numpy()
    chosen = rng.choice(
        entry_ids,
        size=min(n, len(entry_ids)),
        replace=False,
    )

    holiday_list = list(holidays) if holidays else []

    for eid in chosen:
        mask = df["entry_id"] == eid
        idx = df.index[mask][0]
        dt = pd.Timestamp(df.at[idx, "created_at"])

        rand_val = rng.random()
        if rand_val < 0.35:
            # Créer une date de samedi.
            days_to_saturday = (5 - dt.weekday()) % 7
            if days_to_saturday == 0:
                days_to_saturday = 7
            dt = dt + timedelta(days=int(days_to_saturday))
            scenario = "week-end"
        elif rand_val < 0.70 or not holiday_list:
            # Créer une heure nocturne.
            night_hour = int(rng.choice([22, 23, 0, 1, 2, 3, 4, 5]))
            dt = dt.replace(hour=night_hour)
            scenario = "nuit"
        else:
            # Créer une date de jour férié.
            h_str = rng.choice(holiday_list)
            m, d = map(int, h_str.split("-"))
            dt = dt.replace(month=m, day=d)
            scenario = "jour férié"

        # Appliquer la même date de saisie à toutes les lignes
        # de cette écriture comptable.
        df.loc[mask, "created_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")

        ground_truth.append({
            "entry_id": eid,
            "line_id": 0,
            "anomaly_type": "unusual_hours",
            "target_test": "T4",
            "description": (
                f"Saisie atypique ({scenario}) : "
                f"{dt.strftime('%Y-%m-%d %H:%M')}"
            ),
        })

    return df



def _inject_splitting(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_groups: int,
    threshold: float,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T5 : Injecte des groupes d'écritures fractionnées sous le seuil."""

    new_rows = []
    max_entry = int(df["entry_id"].str.extract(r"(\d+)")[0].max())

    for g in range(n_groups):
        user = rng.choice(USERS)
        pair = COMMON_PAIRS[int(rng.integers(0, len(COMMON_PAIRS)))]

        # Créer des montants positifs qui restent tous sous le seuil.
        # Chaque montant doit être assez grand pour être cohérent
        # avec la borne basse utilisée par le test T5.
        lower_bound = threshold * 0.90
        upper_bound = threshold * 0.99

        # Pour ce scénario, deux écritures suffisent à dépasser le seuil.
        # On construit donc des groupes de deux écritures pour lesquels
        # chaque montant est proche du seuil et le total le dépasse.
        first_amount = round(
            float(rng.uniform(lower_bound, upper_bound)), 2
        )
        second_min = max(
            lower_bound,
            round(threshold - first_amount + 0.01, 2),
        )
        second_amount = round(
            float(rng.uniform(second_min, upper_bound)), 2
        )

        amounts = [first_amount, second_amount]

        # Même jour ouvrable pour les deux écritures du groupe.
        doc_date = datetime(2025, 1, 1) + timedelta(
            days=int(rng.integers(0, 365))
        )
        while doc_date.weekday() >= 5:
            doc_date += timedelta(days=1)

        for amt in amounts:
            max_entry += 1
            eid = f"JE-{max_entry:06d}"
            created = _random_business_datetime(rng, doc_date)

            new_rows.append({
                "entry_id": eid,
                "line_id": 1,
                "account": pair[0],
                "account_label": CHART_OF_ACCOUNTS.get(pair[0], ""),
                "debit": amt,
                "credit": 0.0,
                "posting_date": doc_date.strftime("%Y-%m-%d"),
                "document_date": doc_date.strftime("%Y-%m-%d"),
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user,
                "source": "manuelle",
                "description": f"Fractionnement groupe {g + 1}",
            })

            new_rows.append({
                "entry_id": eid,
                "line_id": 2,
                "account": pair[1],
                "account_label": CHART_OF_ACCOUNTS.get(pair[1], ""),
                "debit": 0.0,
                "credit": amt,
                "posting_date": doc_date.strftime("%Y-%m-%d"),
                "document_date": doc_date.strftime("%Y-%m-%d"),
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user,
                "source": "manuelle",
                "description": f"Fractionnement groupe {g + 1}",
            })

            ground_truth.append({
                "entry_id": eid,
                "line_id": 1,
                "anomaly_type": "splitting",
                "target_test": "T5",
                "description": (
                    f"Fractionnement {amt}€, total groupe "
                    f"{round(sum(amounts), 2)}€ > seuil {threshold}€"
                ),
            })

    if new_rows:
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


def _inject_imbalance(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T6 : Injecte des écritures déséquilibrées."""
    entries = df["entry_id"].unique()
    chosen = rng.choice(entries, size=min(n, len(entries)), replace=False)

    for eid in chosen:
        mask = (df["entry_id"] == eid) & (df["debit"] > 0)
        idxs = df[mask].index.tolist()
        if idxs:
            idx = idxs[0]
            offset = round(float(rng.uniform(0.5, 50.0)), 2)
            df.at[idx, "debit"] = round(df.at[idx, "debit"] + offset, 2)
            ground_truth.append({
                "entry_id": eid,
                "line_id": df.at[idx, "line_id"],
                "anomaly_type": "imbalance",
                "target_test": "T6",
                "description": f"Déséquilibre de {offset}€ sur {eid}",
            })

    return df



def _inject_rare_pairs(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T7 : Injecte des paires débit/crédit réellement rares."""
    new_rows = []
    if n <= 0:
        return df

    max_entry = int(df["entry_id"].str.extract(r"(\d+)")[0].max())

    # Recenser les paires débit/crédit déjà présentes.
    debit_accounts = (
        df.loc[df["debit"] > 0, ["entry_id", "account"]]
        .rename(columns={"account": "debit_account"})
    )
    credit_accounts = (
        df.loc[df["credit"] > 0, ["entry_id", "account"]]
        .rename(columns={"account": "credit_account"})
    )
    existing = debit_accounts.merge(credit_accounts, on="entry_id")

    existing_pairs = set(
        zip(
            existing["debit_account"].astype(str),
            existing["credit_account"].astype(str),
        )
    )

    accounts = list(CHART_OF_ACCOUNTS.keys())
    sensitive_prefixes = ("6", "7", "1")

    candidates = [
        (debit, credit)
        for debit in accounts
        for credit in accounts
        if (
            debit != credit
            and (debit, credit) not in existing_pairs
            and (
                str(debit).startswith(sensitive_prefixes)
                or str(credit).startswith(sensitive_prefixes)
            )
        )
    ]

    if len(candidates) < n:
        raise ValueError(
            f"Impossible de créer {n} paires rares distinctes : "
            f"seulement {len(candidates)} combinaisons disponibles."
        )

    chosen_indices = rng.choice(len(candidates), size=n, replace=False)
    unusual_pairs = [candidates[int(i)] for i in chosen_indices]

    for i, pair in enumerate(unusual_pairs):
        max_entry += 1
        eid = f"JE-{max_entry:06d}"
        amount = _generate_amount(rng)
        doc_date = _next_weekday(datetime(2025, 1, 1) + timedelta(
            days=int(rng.integers(0, 365))
        ))
        user = rng.choice(USERS)
        created = _random_business_datetime(rng, doc_date)

        common = {
            "entry_id": eid,
            "posting_date": doc_date.strftime("%Y-%m-%d"),
            "document_date": doc_date.strftime("%Y-%m-%d"),
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user,
            "source": "manuelle",
            "description": f"Paire inhabituelle {pair[0]}-{pair[1]}",
        }

        new_rows.append({
            **common,
            "line_id": 1,
            "account": pair[0],
            "account_label": CHART_OF_ACCOUNTS.get(pair[0], ""),
            "debit": amount,
            "credit": 0.0,
        })
        new_rows.append({
            **common,
            "line_id": 2,
            "account": pair[1],
            "account_label": CHART_OF_ACCOUNTS.get(pair[1], ""),
            "debit": 0.0,
            "credit": amount,
        })

        ground_truth.append({
            "entry_id": eid,
            "line_id": 1,
            "anomaly_type": "rare_pair",
            "target_test": "T7",
            "description": f"Paire inhabituelle {pair[0]} ↔ {pair[1]}",
        })

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def _inject_late_posting(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T8 : Injecte des saisies tardives."""
    entries = df["entry_id"].unique()
    chosen = rng.choice(entries, size=min(n, len(entries)), replace=False)

    for eid in chosen:
        mask = df["entry_id"] == eid
        late_days = int(rng.integers(35, 120))
        idxs = df[mask].index
        for idx in idxs:
            doc = pd.Timestamp(df.at[idx, "document_date"])
            new_posting = doc + timedelta(days=late_days)
            df.at[idx, "posting_date"] = new_posting.strftime("%Y-%m-%d")
        ground_truth.append({
            "entry_id": eid,
            "line_id": 0,
            "anomaly_type": "late_posting",
            "target_test": "T8",
            "description": f"Saisie {late_days} jours après la date du document",
        })

    return df


def _inject_atypical_users(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_users: int,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T9 : Injecte des comportements d'utilisateurs atypiques."""
    new_rows = []
    max_entry = int(df["entry_id"].str.extract(r"(\d+)")[0].max())
    atypical_users = [f"user_atypical_{i}" for i in range(n_users)]

    for user in atypical_users:
        # Cet utilisateur fait beaucoup d'écritures à gros montants
        n_entries = int(rng.integers(200, 400))
        for _ in range(n_entries):
            max_entry += 1
            eid = f"JE-{max_entry:06d}"
            pair = COMMON_PAIRS[int(rng.integers(0, len(COMMON_PAIRS)))]
            # Montants beaucoup plus élevés que la normale
            amount = round(float(rng.uniform(50000, 500000)), 2)
            doc_date = _next_weekday(
                datetime(2025, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
            )
            created = _random_business_datetime(rng, doc_date)

            new_rows.append({
                "entry_id": eid, "line_id": 1,
                "account": pair[0],
                "account_label": CHART_OF_ACCOUNTS.get(pair[0], ""),
                "debit": amount, "credit": 0.0,
                "posting_date": doc_date.strftime("%Y-%m-%d"),
                "document_date": doc_date.strftime("%Y-%m-%d"),
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user, "source": "manuelle",
                "description": f"Écriture utilisateur atypique {user}",
            })
            new_rows.append({
                "entry_id": eid, "line_id": 2,
                "account": pair[1],
                "account_label": CHART_OF_ACCOUNTS.get(pair[1], ""),
                "debit": 0.0, "credit": amount,
                "posting_date": doc_date.strftime("%Y-%m-%d"),
                "document_date": doc_date.strftime("%Y-%m-%d"),
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user, "source": "manuelle",
                "description": f"Écriture utilisateur atypique {user}",
            })
            ground_truth.append({
                "entry_id": eid, "line_id": 1,
                "anomaly_type": "atypical_user",
                "target_test": "T9",
                "description": f"Utilisateur atypique {user} — montant {amount}€",
            })

    if new_rows:
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df



def _inject_benford_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_account: str,
    ground_truth: list[dict],
) -> pd.DataFrame:
    """T1 : Perturbe les montants ciblés sans effacer les déséquilibres T6."""
    mask = df["account"].astype(str) == str(target_account)
    idxs = df.index[mask].tolist()

    count = 0
    processed_entries = set()

    for idx in idxs:
        row = df.loc[idx]
        eid = row["entry_id"]

        # Ne traiter chaque écriture qu'une seule fois.
        if eid in processed_entries:
            continue

        entry = df[df["entry_id"] == eid]
        if entry["description"].astype(str).str.startswith(
              "Fractionnement groupe"
        ).any():
             continue

        # Limiter ce scénario aux écritures simples à deux lignes.
        if len(entry) != 2:
            continue

        other_idx = entry.index[entry.index != idx][0]

        if row["debit"] > 0:
            col = "debit"
            other_col = "credit"
        elif row["credit"] > 0:
            col = "credit"
            other_col = "debit"
        else:
            continue

        # Garder l'écart existant entre débit et crédit.
        old_amount = float(row[col])
        old_counterpart = float(df.at[other_idx, other_col])
        difference = old_amount - old_counterpart

        # Perturber le montant sans imposer artificiellement
        # le même montant aux deux lignes.
        new_amount = round(float(rng.uniform(100, 8000)), 2)
        new_counterpart = round(new_amount - difference, 2)

        if new_counterpart <= 0:
            continue

        df.at[idx, col] = new_amount
        df.at[other_idx, other_col] = new_counterpart

        processed_entries.add(eid)
        count += 1

        ground_truth.append({
            "entry_id": eid,
            "line_id": row["line_id"],
            "anomaly_type": "benford",
            "target_test": "T1",
            "description": (
                f"Montant modifié pour perturber Benford — {new_amount}€"
            ),
        })

    logger.info(
        "T1 Benford : %d écritures modifiées sur le compte %s",
        count,
        target_account,
    )
    return df


def generate_synthetic_data(config_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Génère le jeu de données synthétique complet avec anomalies injectées.

    Args:
        config_dir: Chemin vers le dossier de configuration.
        output_dir: Chemin vers le dossier de sortie.

    Returns:
        Tuple (chemin du CSV grand livre, chemin du CSV vérité terrain).
    """
    thresholds = _load_thresholds(config_dir)
    gen_config = thresholds["data_generation"]
    seed = gen_config["seed"]
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Génération de %d écritures avec seed=%d...", gen_config["n_entries"], seed)

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    holidays = thresholds.get("t4_unusual_hours", {}).get("holidays", [])
    holiday_set = set(holidays)

    # 1. Générer les écritures de base
    df = generate_base_entries(
        rng, gen_config["n_entries"], start_date, end_date, holidays=holiday_set
    )
    logger.info("Écritures de base : %d lignes, %d écritures", len(df), df["entry_id"].nunique())

    # 2. Injecter les anomalies
    ground_truth: list[dict] = []
    injection = gen_config["anomaly_injection"]

    df = _inject_duplicates(df, rng, injection["duplicates"], ground_truth)
    logger.info("Après doublons : %d lignes", len(df))

    df = _inject_round_amounts(
        df, rng, injection["round_amounts"], ground_truth,
        thresholds["t3_round_amounts"]["sensitive_accounts"],
    )

    df = _inject_unusual_hours(
        df, rng, injection["weekend_night"], ground_truth, holidays=holidays
    )

    df = _inject_splitting(
        df, rng, injection["splitting_groups"],
        thresholds["t5_splitting"]["approval_threshold"], ground_truth,
    )

    df = _inject_imbalance(df, rng, injection["imbalanced"], ground_truth)

    df = _inject_rare_pairs(df, rng, injection["rare_pairs"], ground_truth)

    df = _inject_late_posting(df, rng, injection["late_posting"], ground_truth)

    df = _inject_atypical_users(df, rng, injection["atypical_users"], ground_truth)

    # Benford : choisir un compte avec assez de lignes
    account_counts = df["account"].value_counts()
    benford_candidates = account_counts[account_counts >= 300].index.tolist()
    if benford_candidates:
        target_account = benford_candidates[0]
        df = _inject_benford_anomalies(df, rng, target_account, ground_truth)

    # 3. Mélanger et réindexer
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 4. Sauvegarder
    csv_path = output_dir / "general_ledger.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    gt_df = pd.DataFrame(ground_truth)
    gt_path = output_dir / "ground_truth.csv"
    gt_df.to_csv(gt_path, index=False, encoding="utf-8")

    logger.info(
        "Données générées : %d lignes, %d écritures, %d anomalies injectées",
        len(df), df["entry_id"].nunique(), len(gt_df),
    )
    logger.info("Grand livre : %s", csv_path)
    logger.info("Vérité terrain : %s", gt_path)

    return csv_path, gt_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    project_root = Path(__file__).resolve().parent.parent.parent
    generate_synthetic_data(
        config_dir=project_root / "config",
        output_dir=project_root / "data" / "raw",
    )
