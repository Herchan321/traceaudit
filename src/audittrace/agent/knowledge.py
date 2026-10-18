"""
knowledge.py — Base de connaissances normative ISA 240 & RAG.

Indexe et fournit les clauses normatives ISA 240 (Responsabilités de l'auditeur
concernant les fraudes dans un audit d'états financiers) pour justifier
réglementairement chaque anomalie détectée par les tests d'audit.
"""

import re
from dataclasses import dataclass


@dataclass
class ISAReference:
    paragraph: str
    title: str
    content: str
    related_tests: list[str]
    keywords: list[str]


ISA_240_KNOWLEDGE_BASE: list[ISAReference] = [
    ISAReference(
        paragraph="ISA 240.32 (a)",
        title="Test des écritures de journal et ajustements",
        content=(
            "L'auditeur doit concevoir et mettre en œuvre des procédures d'audit pour "
            "tester le caractère approprié des écritures de journal enregistrées dans "
            "le grand livre général et des autres ajustements apportés lors de "
            "l'établissement des états financiers."
        ),
        related_tests=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"],
        keywords=["journal", "écriture", "grand livre", "ajustement", "clôture", "fraude"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (1)",
        title="Écritures sur comptes inhabituels ou rarement mouvementés",
        content=(
            "Les écritures passées sur des comptes inhabituels, rarement utilisés ou non "
            "liés à l'activité courante, ou sur des comptes d'attente (471) présentent un "
            "risque élevé d'anomalie ou de manipulation frauduleuse."
        ),
        related_tests=["T7"],
        keywords=["compte rare", "paires rares", "inhabituel", "attente", "471", "rare_pair"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (2)",
        title="Écritures passées par des personnes non habilitées",
        content=(
            "Les écritures passées par des personnes qui ne passent habituellement pas d'écritures "
            "comptables, ou dont les volumes et profils de saisie s'écartent significativement "
            "de leurs fonctions attribuées, doivent faire l'objet d'un examen approfondi."
        ),
        related_tests=["T9"],
        keywords=["utilisateur", "profil", "droits", "habilitation", "volume", "atypical_user"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (3)",
        title="Écritures enregistrées en fin de période ou avec retard",
        content=(
            "Les écritures enregistrées en fin de période ou après la date de clôture sans "
            "description probante, ou présentant un écart temporel excessif entre la date de pièce "
            "et la date de comptabilisation effective, constituent des indices majeurs d'altération."
        ),
        related_tests=["T8"],
        keywords=["retard", "saisie tardive", "anti-datée", "délai", "late_posting"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (4)",
        title="Écritures saisies en dehors des heures de bureau",
        content=(
            "Les écritures créées ou modifiées en dehors des heures ouvrées normales (heures "
            "nocturnes, week-ends ou jours fériés officiels) présentent un risque accru d'évitement "
            "des contrôles hiérarchiques et de collusion."
        ),
        related_tests=["T4"],
        keywords=["horaires", "nuit", "week-end", "dimanche", "jour férié", "unusual_hours"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (5)",
        title="Montants ronds et anomalies de distribution de chiffres",
        content=(
            "Les écritures comportant des montants parfaitement ronds (ex: multiples de 1000 €) "
            "ou s'écartant de la distribution naturelle logarithmique (Loi de Benford) sur les "
            "premiers chiffres significatifs indiquent fréquemment des montants estimés ou inventés."
        ),
        related_tests=["T1", "T3"],
        keywords=["montant rond", "chiffre", "benford", "mad", "estimation", "round_amount"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (6)",
        title="Fractionnement de montants sous les seuils d'autorisation",
        content=(
            "Le découpage d'opérations en plusieurs écritures successives d'un montant immédiatement "
            "inférieur aux seuils de délégation de signature ou d'approbation managériale (smurfing) "
            "caractérise une volonté manifeste de contournement du contrôle interne."
        ),
        related_tests=["T5"],
        keywords=["fractionnement", "seuil", "contournement", "approbation", "splitting", "smurfing"],
    ),
    ISAReference(
        paragraph="ISA 240.A43 (7)",
        title="Écritures déséquilibrées et doublons",
        content=(
            "Toute rupture du principe fondamental de la partie double (Débit ≠ Crédit) ou la présence "
            "d'écritures en double (même montant, même date, même tiers) traduit soit une défaillance grave "
            "du système comptable, soit un risque de double décaissement ou détournement d'actifs."
        ),
        related_tests=["T2", "T6"],
        keywords=["déséquilibre", "débit", "crédit", "doublon", "imbalance", "duplicate"],
    ),
]


def search_isa_norm(query: str, top_k: int = 3) -> list[dict]:
    """
    Recherche les clauses pertinentes d'ISA 240 pour une requête ou un code de test.
    Utilise un score pondéré combinant correspondance de tests et pertinence lexicale.
    """
    query_lower = query.lower()
    scored = []

    for item in ISA_240_KNOWLEDGE_BASE:
        score = 0
        # 1. Correspondance directe sur les identifiants de test (ex: T1, T4)
        for t in item.related_tests:
            if re.search(rf"\b{t}\b", query, re.IGNORECASE):
                score += 5

        # 2. Correspondance sur les mots-clés
        for kw in item.keywords:
            if kw.lower() in query_lower:
                score += 2

        # 3. Correspondance textuelle
        words = query_lower.split()
        for w in words:
            if len(w) >= 4:
                if w in item.title.lower():
                    score += 1.5
                if w in item.content.lower():
                    score += 0.5

        if score > 0:
            scored.append((score, item))

    # Trier par score décroissant
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for s, item in scored[:top_k]:
        results.append({
            "paragraph": item.paragraph,
            "title": item.title,
            "content": item.content,
            "relevance_score": round(s, 2),
            "related_tests": item.related_tests,
        })

    # Si aucun match direct, renvoyer la clause générale ISA 240.32
    if not results:
        general = ISA_240_KNOWLEDGE_BASE[0]
        results.append({
            "paragraph": general.paragraph,
            "title": general.title,
            "content": general.content,
            "relevance_score": 1.0,
            "related_tests": general.related_tests,
        })

    return results
