"""
t4_unusual_hours.py — Test des horaires atypiques (T4).

Détecte les écritures saisies en dehors des heures de bureau :
week-end, jours fériés, ou entre 22h et 6h.

Implémentation en Python.
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

WEEKDAY_NAMES = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _build_holiday_dates(holidays_config: list[str], year: int = 2025) -> set[str]:
    """Construit l'ensemble des dates de jours fériés à partir de la config MM-DD."""
    dates = set()
    for h in holidays_config:
        try:
            month, day = h.split("-")
            dt = datetime(year, int(month), int(day))
            dates.add(dt.strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            continue
    return dates


def run_t4_unusual_hours(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Détecte les saisies à des horaires atypiques.

    Args:
        df: Grand livre nettoyé.
        config: Section t4_unusual_hours du fichier thresholds.yaml.

    Returns:
        DataFrame des résultats au format standard.
    """
    night_start = config.get("night_start", 22)
    night_end = config.get("night_end", 6)
    holidays_raw = config.get("holidays", [])

    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work = work.dropna(subset=["created_at"])

    # Extraire les composantes temporelles
    work["hour"] = work["created_at"].dt.hour
    work["weekday"] = work["created_at"].dt.weekday  # 0=lundi, 6=dimanche
    work["date_str"] = work["created_at"].dt.strftime("%Y-%m-%d")

    # Construire les dates de jours fériés pour toutes les années présentes
    years = work["created_at"].dt.year.unique()
    holidays = set()
    for year in years:
        holidays.update(_build_holiday_dates(holidays_raw, year))

    # Conditions de saisie atypique
    is_weekend = work["weekday"] >= 5
    is_night = (work["hour"] >= night_start) | (work["hour"] < night_end)
    is_holiday = work["date_str"].isin(holidays)

    flagged = work[is_weekend | is_night | is_holiday].drop_duplicates(subset=["entry_id"])

    results = []
    for _, row in flagged.iterrows():
        reasons = []
        if row["weekday"] >= 5:
            day_name = WEEKDAY_NAMES[row["weekday"]]
            reasons.append(f"week-end ({day_name})")
        if row["hour"] >= night_start or row["hour"] < night_end:
            reasons.append(f"heure nocturne ({row['hour']}h)")
        if row["date_str"] in holidays:
            reasons.append("jour férié")

        reason_str = ", ".join(reasons)

        results.append({
            "entry_id": row["entry_id"],
            "test_id": "T4",
            "observed_value": (
                f"heure={row['hour']}h, "
                f"jour={row['weekday']}, "
                f"date={row['date_str']}"
            ),
            "threshold": (
                f"nuit=[{night_start}h-{night_end}h], "
                f"week-end, jours_fériés={len(holidays)}"
            ),
            "message": (
                f"Saisie atypique : {reason_str} — "
                f"écriture créée le {row['date_str']} à {row['hour']}h"
                f" par {row.get('user_id', 'N/A')}"
            ),
        })

    result_df = pd.DataFrame(results)
    logger.info("T4 Horaires atypiques : %d signalements", len(result_df))
    return result_df
