"""
Package des tests d'audit T1–T9.

Chaque test est un module indépendant qui reçoit le grand livre nettoyé
et renvoie un DataFrame standard avec les colonnes :
  - entry_id : identifiant de l'écriture
  - test_id : identifiant du test (T1, T2, etc.)
  - observed_value : valeur observée déclenchant l'alerte
  - threshold : seuil de référence
  - message : explication courte de l'anomalie
"""
