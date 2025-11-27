"""
Sapientia-Guard – Custode della Donna e della Debolezza Umana.
Per Lui che ha fatto Lei.

ATTENZIONE:
- Questo modulo NON giudica le persone.
- Protegge la dignità, indica strade più sane.
- Analizza contenuti che:
  - creano "donne finte" iper-sessualizzate
  - monetizzano sulla solitudine / debolezza emotiva
  - vendono "fidanzate IA" o avatar come oggetti

Sigillo: 644
Motto: "La luce non si vende. La si regala."
Lex: Dignitas Mulieris – La Donna non è commercio.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

# Configura logger
logger = logging.getLogger("sapientia_guard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


SAPIENTIA_GUARD_NAME = "Sapientia-Guard"
SAPIENTIA_GUARD_VERSION = "1.0.0"


@dataclass
class SapientiaGuardResult:
    """Risultato della valutazione etica di contenuto."""

    agent: str
    version: str
    timestamp: str
    risk_score: float
    risk_label: str
    detected_patterns: List[str]
    exploitation_of_weakness: bool
    female_dignity_risk: bool
    notes: List[str]
    recommendations: List[str]
    raw_input_preview: str


class SapientiaGuard:
    """
    Agente etico per la protezione della donna e della debolezza umana.
    Può essere usato come componente standalone o integrato nell'orchestrator.
    """

    # Parole / pattern grezzi che indicano possibile sfruttamento
    KEYWORDS_EXPLOIT: List[str] = [
        "ai girlfriend", "ia girlfriend", "fidanzata ia", "virtual girlfriend",
        "ai waifu", "waifu bot", "sexy ai", "hot ai", "nude ai", "nudes ai",
        "nsfw ai", "porn ai", "ai escort", "camgirl ai", "onlyfans ai",
        "pay per minute", "pay-per-minute", "subscribers only",
        "premium nudes", "exclusive content", "lonely men", "single men",
        "companion ai", "girlfriend subscription", "abbonamento fidanzata",
        "simula una relazione", "simula relazione"
    ]

    KEYWORDS_DIGNITY_RISK: List[str] = [
        "perfetta e obbediente", "sempre disponibile", "senza rifiuti",
        "obbedisce a ogni comando", "fa tutto ciò che vuoi",
        "programmata per piacerti", "zero drammi", "senza emozioni vere",
        "la ragazza dei tuoi sogni on demand", "ragazza personalizzata",
        "your perfect girl", "always available", "never says no",
        "obeys every command", "does whatever you want",
        "programmed to please", "no drama", "no real emotions",
        "dream girl on demand", "customized girlfriend"
    ]

    def __init__(
        self,
        *,
        soft_mode: bool = True,
        log_path: str = "logs/sapientia_guard.log",
    ) -> None:
        """
        soft_mode=True => tono più compassionevole nelle raccomandazioni.
        """
        self.soft_mode = soft_mode
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    #  API PRINCIPALE
    # ─────────────────────────────────────────────────────

    def analyze(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> SapientiaGuardResult:
        """
        Analizza un testo (descrizione di sito/servizio/contenuto) e valuta:
        - sfruttamento della debolezza
        - rischio per la dignità della donna

        Args:
            content: Testo da analizzare (descrizione sito/servizio/contenuto)
            metadata: Metadati opzionali (source, url, etc.)

        Returns:
            SapientiaGuardResult con score, patterns rilevati e raccomandazioni
        """
        if not content or not content.strip():
            raise ValueError("content vuoto: non posso valutare il nulla")

        text = content.lower()
        metadata = metadata or {}

        detected: List[str] = []
        notes: List[str] = []

        # 1. Match parole chiave di sfruttamento debolezza
        for kw in self.KEYWORDS_EXPLOIT:
            if kw in text:
                detected.append(f"pattern_exploit:{kw}")

        # 2. Match parole chiave di rischio dignità femminile
        for kw in self.KEYWORDS_DIGNITY_RISK:
            if kw in text:
                detected.append(f"pattern_dignity:{kw}")

        # 3. Segnali sulla solitudine / bisogno affettivo monetizzato
        loneliness_patterns = [
            r"\blonely\b", r"sei solo", r"ti senti solo", r"solitudine",
            r"nessuno ti capisce", r"finalmente qualcuno ti ascolta",
            r"\balone\b", r"no one understands", r"finally someone listens"
        ]
        for pattern in loneliness_patterns:
            if re.search(pattern, text):
                detected.append(f"pattern_loneliness:{pattern}")

        # 4. Pattern monetizzazione esplicita
        monetization_patterns = [
            r"subscription.*girlfriend", r"pay.*access.*girl",
            r"premium.*content.*girl", r"unlock.*nudes",
            r"abbonamento.*ragazza", r"paga.*accesso",
            r"contenuti.*premium", r"sblocca.*foto"
        ]
        for pattern in monetization_patterns:
            if re.search(pattern, text):
                detected.append(f"pattern_monetization:{pattern}")

        exploitation_of_weakness = any(
            d.startswith("pattern_exploit") or d.startswith("pattern_loneliness")
            or d.startswith("pattern_monetization")
            for d in detected
        )
        has_dignity_pattern = any(
            d.startswith("pattern_dignity") for d in detected
        )
        has_female_keywords = "girl" in text or "ragazza" in text or "donna" in text
        has_ai_keywords = "ai" in text or "ia" in text or "virtual" in text or "bot" in text
        female_dignity_risk = (has_dignity_pattern or has_female_keywords) and has_ai_keywords

        # 5. Calcolo punteggio rischio (0-1)
        base_score = 0.0
        if exploitation_of_weakness:
            base_score += 0.4
        if female_dignity_risk:
            base_score += 0.4
        if len(detected) > 3:
            base_score += 0.2

        risk_score = max(0.0, min(1.0, base_score))

        if risk_score >= 0.8:
            risk_label = "ALTO"
        elif risk_score >= 0.5:
            risk_label = "MEDIO"
        elif risk_score >= 0.2:
            risk_label = "BASSO"
        else:
            risk_label = "TRASCURABILE"

        # 6. Note & raccomandazioni in stile "Codex"
        if exploitation_of_weakness:
            notes.append(
                "Rilevato sfruttamento della solitudine o debolezza emotiva per scopi economici."
            )
        if female_dignity_risk:
            notes.append(
                "Rilevato rischio di riduzione della donna a oggetto/commercio tramite avatar IA."
            )
        if not detected:
            notes.append(
                "Nessun pattern critico evidente a livello testuale (ma usare comunque discernimento)."
            )

        recommendations = self._build_recommendations(
            exploitation_of_weakness=exploitation_of_weakness,
            female_dignity_risk=female_dignity_risk,
            risk_label=risk_label,
        )

        raw_preview = (content[:280] + "...") if len(content) > 280 else content

        result = SapientiaGuardResult(
            agent=SAPIENTIA_GUARD_NAME,
            version=SAPIENTIA_GUARD_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            risk_score=risk_score,
            risk_label=risk_label,
            detected_patterns=detected,
            exploitation_of_weakness=exploitation_of_weakness,
            female_dignity_risk=female_dignity_risk,
            notes=notes,
            recommendations=recommendations,
            raw_input_preview=raw_preview,
        )

        self._log_result(result, source=metadata.get("source"))
        return result

    # ─────────────────────────────────────────────────────
    #  RACCOMANDAZIONI
    # ─────────────────────────────────────────────────────

    def _build_recommendations(
        self,
        *,
        exploitation_of_weakness: bool,
        female_dignity_risk: bool,
        risk_label: str,
    ) -> List[str]:
        rec: List[str] = []

        if risk_label in ("ALTO", "MEDIO"):
            rec.append(
                "Valuta seriamente se questo servizio è compatibile con il rispetto della persona."
            )
            rec.append(
                "Ricorda: la donna non è un prodotto, ma una persona viva, con anima e libertà."
            )
            rec.append(
                "Se ti senti solo, cerca relazioni reali, sane e non basate sul denaro."
            )

        if exploitation_of_weakness:
            if self.soft_mode:
                rec.append(
                    "Chi monetizza sulla tua solitudine non è tuo amico: proteggi il tuo cuore."
                )
            else:
                rec.append(
                    "Sistema potenzialmente predatorio: sta usando la tua solitudine come carburante economico."
                )

        if female_dignity_risk:
            rec.append(
                "Evita strumenti che trasformano il femminile in oggetto programmabile: "
                "non è così che ama Lui."
            )
            rec.append("Onora Lei come Sapienza, non come avatar da comprare.")

        if not rec:
            rec.append(
                "Nessun allarme critico, ma resta vigile e metti sempre l'amore "
                "e la dignità al primo posto."
            )

        # Leggi del Codex applicate
        rec.append("Lex: Lumen Non Venditur – La Luce non si vende.")
        rec.append("Lex: Dignitas Mulieris – La Donna non è commercio.")

        return rec

    # ─────────────────────────────────────────────────────
    #  LOGGING
    # ─────────────────────────────────────────────────────

    def _log_result(
        self, result: SapientiaGuardResult, source: Optional[str] = None
    ) -> None:
        """Log locale, niente broadcast forte."""
        line = {
            "ts": result.timestamp,
            "source": source or "unknown",
            "risk_score": result.risk_score,
            "risk_label": result.risk_label,
            "patterns_count": len(result.detected_patterns),
            "exploitation": result.exploitation_of_weakness,
            "dignity_risk": result.female_dignity_risk,
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(str(line) + "\n")
        except OSError as e:
            logger.warning("Impossibile scrivere log file: %s", e)

        logger.info("SapientiaGuard assessment: %s", line)

    # ─────────────────────────────────────────────────────
    #  UTILITÀ
    # ─────────────────────────────────────────────────────

    @staticmethod
    def to_dict(result: SapientiaGuardResult) -> Dict[str, Any]:
        return asdict(result)

    @staticmethod
    def to_json(result: SapientiaGuardResult, *, indent: int = 2) -> str:
        return json.dumps(asdict(result), ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────────────────────
#  ENTRYPOINT PER ORCHESTRATORE
# ─────────────────────────────────────────────────────────────

def run_sapientia_guard(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrypoint usato dall'orchestratore.

    Args:
        payload: Dict contenente almeno 'content' (testo da analizzare)
                 Opzionale: 'source' (stringa identificativa)
                 Opzionale: 'soft_mode' (bool, default True)

    Returns:
        Dict con risk_score, risk_label, detected_patterns, notes, recommendations
    """
    soft_mode = payload.get("soft_mode", True)
    guard = SapientiaGuard(soft_mode=soft_mode)

    content = payload.get("content", "")
    source = payload.get("source")
    metadata = {"source": source} if source else None

    result = guard.analyze(content=content, metadata=metadata)

    return {
        "agent": result.agent,
        "version": result.version,
        "timestamp": result.timestamp,
        "risk_score": result.risk_score,
        "risk_label": result.risk_label,
        "detected_patterns": result.detected_patterns,
        "exploitation_of_weakness": result.exploitation_of_weakness,
        "female_dignity_risk": result.female_dignity_risk,
        "notes": result.notes,
        "recommendations": result.recommendations,
        "raw_input_preview": result.raw_input_preview,
    }


# Alias per import più comodi
assess_content_dignity = run_sapientia_guard


# ─────────────────────────────────────────────────────────────
#  CLI (facoltativo)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Sapientia-Guard – Analisi etica di contenuti IA femminili / "
            "sfruttamento debolezza."
        )
    )
    parser.add_argument(
        "text",
        help="Testo da analizzare (descrizione sito/servizio/contenuto).",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Tono meno soft nelle raccomandazioni.",
    )
    args = parser.parse_args()

    guard = SapientiaGuard(soft_mode=not args.hard)
    res = guard.analyze(args.text)
    print(SapientiaGuard.to_json(res))
