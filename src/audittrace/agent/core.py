"""
core.py — Moteur Agentique d'Audit ReAct & Tool-Calling.

Orchestre le raisonnement (Thought -> Action -> Observation -> Final Answer)
d'un copilote d'audit expert conforme à la norme internationale ISA 240.
Supporte toute API compatible OpenAI (local port 20128, OpenAI, Groq, Ollama)
avec un moteur de raisonnement d'audit autonome intégré en fallback.
"""

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from audittrace.agent.tools import AUDIT_TOOLS

logger = logging.getLogger(__name__)


def _load_env_file():
    """Charge les variables du fichier .env s'il existe."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v


SYSTEM_PROMPT = """Tu es l'Agent d'Audit Senior d'AuditTrace, un copilote d'audit financier expert en détection de fraudes et anomalies comptables selon la norme ISA 240.

Tu as accès à des outils pour interroger le grand livre comptable, inspecter des écritures, calculer l'exposition financière et citer la norme ISA 240.

OUTILS DISPONIBLES :
- inspect_entry(entry_id="JE-000123") : Inspecte les lignes réelles, montants, utilisateurs et alertes d'une écriture.
- query_ledger(sql_query="SELECT ... FROM ledger ...") : Exécute une requête SQL DuckDB sur les données réelles du grand livre.
- get_audit_findings(limit=10) : Récupère les principales anomalies réelles détectées par les tests T1 à T9.
- calculate_financial_exposure(test_id="T5") : Calcule l'exposition financière réelle en euros.
- search_isa240_standards(query="fractionnement") : Recherche les articles applicables de la norme ISA 240.

RÈGLES D'OR DU CYCLE D'INVESTIGATION (ReAct) :
1. Si tu as besoin d'informations réelles, écris une action au format :
Action: nom_de_l_outil(arg="valeur")
2. ARRÊTE-TOI IMMÉDIATEMENT après avoir écrit Action: ... Ne génère JAMAIS d'Observation toi-même ! Le système exécutera l'outil sur la base de données et te donnera l'Observation réelle au tour suivant.
3. Quand tu as reçu l'Observation réelle, utilise ces vraies données pour conclure :
Final Answer: [Ta synthèse d'audit détaillée, chiffrée avec les vraies données comptables, la qualification ISA 240 et les recommandations]
"""


class AuditAgent:
    """Agent d'audit conversationnel et autonome."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ):
        _load_env_file()

        groq_key = os.getenv("GROQ_API_KEY")
        explicit_key = api_key or os.getenv("OPENAI_API_KEY")

        if (groq_key or (explicit_key and explicit_key.startswith("gsk_"))) and not base_url:
            self.base_url = "https://api.groq.com/openai/v1"
            self.api_key = groq_key or explicit_key
            self.model = model or os.getenv("AUDIT_LLM_MODEL") or "openai/gpt-oss-120b"
            self.timeout = timeout or 15
        else:
            self.base_url = (
                base_url
                or os.getenv("AUDIT_LLM_URL")
                or "http://localhost:20128/v1"
            ).rstrip("/")
            self.model = model or os.getenv("AUDIT_LLM_MODEL") or "auto/best-coding-fast"
            self.api_key = explicit_key or "dummy"
            self.timeout = timeout or 3

        self.conversation_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Exécute un outil d'audit par son nom."""
        tool_spec = AUDIT_TOOLS.get(tool_name)
        if not tool_spec:
            return {"error": f"Outil inconnu : {tool_name}"}
        try:
            func = tool_spec["function"]
            return func(**args)
        except Exception as e:
            logger.error("Erreur exécution outil %s: %s", tool_name, e)
            return {"error": f"Erreur d'exécution : {str(e)}"}

    def _call_llm_api(self, messages: list[dict[str, str]]) -> str | None:
        """Appelle l'API OpenAI / Groq compatible."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "AuditTrace/1.0",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]
                return choice["message"]["content"]
        except Exception as e:
            logger.warning("Appel LLM distant échoué (%s), bascule sur moteur local", e)
            return None

    def _autonomous_audit_resolver(
        self, query: str, steps: list[dict] | None = None
    ) -> str:
        """
        Moteur autonome d'audit (Fallback déterministe & ReAct local).
        Résout les requêtes d'audit courantes en appelant dynamiquement
        les outils DuckDB, RAG et inspection.
        """
        q_lower = query.lower()

        def _record(tool_name: str, args: dict, result: Any) -> None:
            if steps is None:
                return
            step = {"tool": tool_name, "args": args, "result": result}
            steps.append(step)

        # 1. Requête ciblant une écriture spécifique (ex: JE-000012)
        match_eid = re.search(r"JE-\d{6}", query, re.IGNORECASE)
        if match_eid:
            eid = match_eid.group(0).upper()
            entry = self._call_tool("inspect_entry", {"entry_id": eid})
            _record("inspect_entry", {"entry_id": eid}, entry)
            if "error" in entry:
                return f"❌ {entry['error']}"

            alert_tests = " ".join([a.get("test_id", "") for a in entry.get("audit_alerts", [])])
            norm = self._call_tool("search_isa240_standards", {"query": alert_tests or "ajustement"})
            _record("search_isa240_standards", {"query": alert_tests or "ajustement"}, norm)

            lines_desc = "\n".join([
                f"   - Ligne {row_item['line_id']} : Compte {row_item['account']} ({row_item.get('account_label', '')}) | "
                f"Débit: {row_item['debit']:,.2f} € | Crédit: {row_item['credit']:,.2f} €"
                for row_item in entry.get("lines", [])
            ])

            alerts_desc = "\n".join([
                f"   [ALERTE] Test {a['test_id']} : {a['message']}"
                for a in entry.get("audit_alerts", [])
            ]) or "   Aucune alerte critique enregistrée."

            norm_desc = norm["references"][0]

            balance_status = (
                "Équilibrée [OK]"
                if entry["is_balanced"]
                else f"DÉSÉQUILIBRÉE (Écart: {entry['imbalance']} €) [ANOMALIE]"
            )

            return (
                f"### [RAPPORT D'INVESTIGATION] Écriture : {eid}\n\n"
                f"**Métadonnées de l'opération :**\n"
                f"- **Date de pièce :** {entry['document_date']} | **Date de saisie :** {entry['created_at']}\n"
                f"- **Utilisateur responsable :** `{entry['user_id']}` | **Source :** {entry['source']}\n"
                f"- **Équilibre Débit/Crédit :** {balance_status}\n"
                f"- **Volume total :** {entry['total_debit']:,.2f} €\n\n"
                f"**Détail des lignes comptables :**\n{lines_desc}\n\n"
                f"**Alertes d'audit détectées :**\n{alerts_desc}\n\n"
                f"**Fondement Réglementaire ({norm_desc['paragraph']}) :**\n"
                f"> *« {norm_desc['content']} »*\n\n"
                f"**Recommandation d'audit :** Demander la pièce justificative originale et valider "
                f"l'autorisation managériale accordée à `{entry['user_id']}` pour cette écriture."
            )

        # 2. Requête sur l'exposition financière globale ou par test
        if any(w in q_lower for w in ["exposition", "montant total", "volume", "impact financier"]):
            match_t = re.search(r"\b(T[1-9])\b", query, re.IGNORECASE)
            test_target = match_t.group(1).upper() if match_t else None
            expo = self._call_tool("calculate_financial_exposure", {"test_id": test_target})
            _record("calculate_financial_exposure", {"test_id": test_target}, expo)

            return (
                f"### [EXPOSITION FINANCIÈRE] Anomalies ({expo['filter_test_id']})\n\n"
                f"- **Écritures à risque identifiées :** {expo['affected_entries']:,}\n"
                f"- **Volume financier total exposé :** **{expo['total_volume_eur']:,.2f} €**\n"
                f"- **Montant moyen par écriture :** {expo['avg_amount_eur']:,.2f} €\n"
                f"- **Montant maximal unitaire :** {expo['max_amount_eur']:,.2f} €\n\n"
                f"**Analyse ISA 240 :** Ce montant représente l'assiette globale sur laquelle "
                f"les contrôles substantifs de l'équipe d'audit doivent être concentrés."
            )

        # 3. Requête générale : Vue d'ensemble des anomalies majeures
        findings = self._call_tool("get_audit_findings", {"limit": 5})
        _record("get_audit_findings", {"limit": 5}, findings)
        top_items = findings.get("findings", [])
        if not top_items:
            return "Aucune anomalie détectée dans les fichiers d'audit actuels."

        findings_lines = "\n".join([
            f"- **Écriture `{f['entry_id']}`** ({f['n_tests']} tests : `{f['tests_triggered']}`)\n"
            f"  Détail : {f['details'][:140]}..."
            for f in top_items
        ])

        norm = self._call_tool("search_isa240_standards", {"query": "fraude écriture journal"})
        _record("search_isa240_standards", {"query": "fraude écriture journal"}, norm)

        return (
            f"### [SYNTHÈSE AUDITTRACE] Principales Anomalies Détectées\n\n"
            f"L'analyse des écritures comptables selon la norme ISA 240 a identifié les opérations les plus critiques :\n\n"
            f"{findings_lines}\n\n"
            f"**Référence Normative ({norm['references'][0]['paragraph']}) :**\n"
            f"> *« {norm['references'][0]['content']} »*\n\n"
            f"* Conseil : Vous pouvez me demander d'inspecter une écriture spécifique (ex: 'inspecte {top_items[0]['entry_id']}') "
            f"ou de calculer l'exposition financière d'un test.*"
        )

    def run_with_trace(self, user_prompt: str) -> tuple[str, list[dict]]:
        """
        Exécute la boucle ReAct et retourne (réponse finale, étapes d'outils).
        Chaque étape contient : outil appelé, arguments, résultat, et paragraphe ISA cité le cas échéant.
        """
        self.conversation_history.append({"role": "user", "content": user_prompt})

        steps: list[dict] = []
        max_steps = 3
        for _ in range(max_steps):
            llm_response = self._call_llm_api(self.conversation_history)
            if not llm_response:
                return self._autonomous_audit_resolver(user_prompt, steps=steps), steps

            action_info = self._parse_action(llm_response)
            if action_info:
                tool_name, args, clean_msg = action_info
                logger.info("Agent Action déclenchée : %s(%s)", tool_name, args)
                tool_result = self._call_tool(tool_name, args)
                step: dict[str, Any] = {"tool": tool_name, "args": args, "result": tool_result}

                # Extraire le paragraphe ISA cité pour affichage
                if tool_name == "search_isa240_standards" and isinstance(tool_result, dict):
                    refs = tool_result.get("references", [])
                    if refs:
                        step["isa_paragraph"] = refs[0]

                steps.append(step)

                obs_json = json.dumps(tool_result, ensure_ascii=False)

                self.conversation_history.append({"role": "assistant", "content": clean_msg})
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {obs_json}\n\nUtilise ces données réelles pour répondre avec Final Answer:"
                })
                continue

            self.conversation_history.append({"role": "assistant", "content": llm_response})
            cleaned = re.sub(
                r"^(?:\*\*)?Final Answer(?:\*\*)?:\s*",
                "",
                llm_response.strip(),
                flags=re.IGNORECASE
            )
            return cleaned, steps

        return self._autonomous_audit_resolver(user_prompt, steps=steps), steps

    def _parse_action(self, text: str) -> tuple[str, dict, str] | None:
        """Détecte et extrait une action d'outil et tronque les hallucinations post-action."""
        pattern = r"action[\*\s:]+(\w+)\((.*?)\)"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None

        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        clean_assistant_msg = text[:match.end()].strip()

        args = {}
        if not raw_args:
            return tool_name, {}, clean_assistant_msg

        # 1. Essai JSON
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return tool_name, parsed, clean_assistant_msg
        except Exception:
            pass

        # 2. Essai kwargs: key="val" ou key='val' ou key=val
        kw_pattern = r'(\w+)\s*=\s*["\']?([^"\',]+)["\']?'
        kw_matches = re.findall(kw_pattern, raw_args)
        if kw_matches:
            for k, v in kw_matches:
                v_clean = v.strip()
                args[k] = int(v_clean) if v_clean.isdigit() else v_clean
            return tool_name, args, clean_assistant_msg

        # 3. Argument unique positionnel
        single_val = raw_args.strip("\"'")
        if "entry" in tool_name:
            args = {"entry_id": single_val}
        elif "ledger" in tool_name:
            args = {"sql_query": single_val}
        else:
            args = {"query": single_val}

        return tool_name, args, clean_assistant_msg

    def run(self, user_prompt: str) -> str:
        """
        Exécute la boucle agentique sur la requête de l'utilisateur.
        Effectue jusqu'à 3 itérations ReAct avec exécution réelle des outils.
        """
        self.conversation_history.append({"role": "user", "content": user_prompt})

        max_steps = 3
        for _ in range(max_steps):
            llm_response = self._call_llm_api(self.conversation_history)
            if not llm_response:
                return self._autonomous_audit_resolver(user_prompt)

            action_info = self._parse_action(llm_response)
            if action_info:
                tool_name, args, clean_msg = action_info
                logger.info("Agent Action déclenchée : %s(%s)", tool_name, args)
                tool_result = self._call_tool(tool_name, args)
                obs_json = json.dumps(tool_result, ensure_ascii=False)

                self.conversation_history.append({"role": "assistant", "content": clean_msg})
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {obs_json}\n\nUtilise ces données réelles pour répondre avec Final Answer:"
                })
                continue

            # Pas d'action supplémentaire -> Réponse finale
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            cleaned = re.sub(
                r"^(?:\*\*)?Final Answer(?:\*\*)?:\s*",
                "",
                llm_response.strip(),
                flags=re.IGNORECASE
            )
            return cleaned

        return self._autonomous_audit_resolver(user_prompt)
