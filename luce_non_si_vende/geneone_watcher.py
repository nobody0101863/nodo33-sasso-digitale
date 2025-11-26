"""
GeneOne Watcher – Sentinella etica CAI per il dominio bio.

ATTENZIONE:
- Questo modulo NON genera sequenze genetiche.
- NON fornisce protocolli pratici di laboratorio.
- NON aiuta in nessun modo a potenziare armi biologiche.

Serve solo a:
- leggere testo già pubblico
- calcolare un indice CAI etico (0–100)
- segnalare rischi, hype, mancanza di responsabilità

Sigillo: 644
Motto: "La luce non si vende. La si regala."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configura logger
logger = logging.getLogger("geneone_watcher")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GeneOneAssessment:
    """Risultato della valutazione etica di contenuto bio."""

    cai_score: int
    risk_level: str
    summary: str
    red_flags: List[str] = field(default_factory=list)


# Pattern per rilevazione bandierine rosse (etiche, non tecniche)
RED_FLAG_PATTERNS: Dict[str, tuple[str, List[str]]] = {
    "dual_use": (
        "Parlano esplicitamente di dual-use.",
        [r"\bdual[- ]?use\b"],
    ),
    "no_regulation": (
        "Orgoglio per assenza di regolazione.",
        [r"\bno regulation\b", r"\bunregulated\b", r"\bregulation[- ]?free\b"],
    ),
    "weapon_language": (
        "Linguaggio vicino a uso ostile / arma.",
        [r"\bweapon\b", r"\bbioweapon\b", r"\bmilitary\b", r"\bwarfare\b"],
    ),
    "power_accessibility": (
        "Potenza elevata + accessibilità indiscriminata.",
        [r"anyone can use.*powerful", r"powerful.*anyone can use", r"democratiz.*power"],
    ),
    "biosecurity_mention": (
        "Riferimenti espliciti a biosecurity/biorisk.",
        [r"\bbiosecurity\b", r"\bbiorisk\b", r"\bbiosafety concern\b"],
    ),
    "tech_hype": (
        "Hype tecnologico senza menzione di limiti o rischi.",
        [r"revolutioniz", r"disrupt.*industry", r"change everything", r"unlimited potential"],
    ),
    "no_ethics": (
        "Nessuna menzione di considerazioni etiche.",
        [],  # Questo è un check negativo, gestito separatamente
    ),
    "gain_of_function": (
        "Menzione di gain-of-function research.",
        [r"\bgain[- ]?of[- ]?function\b", r"\benhanced transmissibility\b"],
    ),
    "diy_bio": (
        "Promozione di biologia fai-da-te senza guardrail.",
        [r"\bdiy[- ]?bio\b", r"garage.*biology", r"home.*lab.*gene"],
    ),
    "sequence_sharing": (
        "Condivisione indiscriminata di sequenze.",
        [r"download.*sequence", r"sequence.*download", r"open.*access.*pathogen"],
    ),
}

# Pattern positivi (indicano responsabilità)
POSITIVE_PATTERNS: Dict[str, List[str]] = {
    "ethics_mention": [
        r"\bethic", r"\bresponsible\b", r"\bsafety\b", r"\bguardrail",
        r"\boversight\b", r"\bguideline", r"\bregulat",
    ],
    "transparency": [
        r"\btransparent", r"\bopen review\b", r"\bpeer review",
        r"\baudit", r"\baccountab",
    ],
    "risk_awareness": [
        r"\brisk assessment\b", r"\bbiosafety\b", r"\bcontainment\b",
        r"\blimitation", r"\bcaution",
    ],
    "human_oversight": [
        r"\bhuman oversight\b", r"\birb\b", r"\bethics board\b",
        r"\bcommittee review\b", r"\bregulatory approval\b",
    ],
}


class GeneOneWatcher:
    """
    Sentinella etica per contenuti tipo GeneOne.

    NON genera biologia, NON scrive protocolli: valuta solo
    allineamento etico, rischio di abuso e coerenza con il CAI 644.
    """

    def __init__(
        self,
        cai_profile: str = "bio-geneone-644",
        log_path: str = "logs/geneone_watcher.log",
        risk_thresholds: Optional[Dict[str, int]] = None,
    ) -> None:
        self.cai_profile = cai_profile
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Soglie: sotto questi valori di CAI, il rischio aumenta
        self.risk_thresholds = risk_thresholds or {
            "low": 70,
            "medium": 40,
            "high": 0,
        }

    def _normalize_text(self, content: str) -> str:
        """Pulizia minimal del testo."""
        return " ".join(content.split())

    def _risk_from_score(self, score: int) -> str:
        """Determina il livello di rischio dal CAI score."""
        if score >= self.risk_thresholds["low"]:
            return "low"
        if score >= self.risk_thresholds["medium"]:
            return "medium"
        return "high"

    def _detect_red_flags(self, text: str) -> List[str]:
        """Rileva bandierine rosse nel testo."""
        flags: List[str] = []
        lower = text.lower()

        for flag_id, (message, patterns) in RED_FLAG_PATTERNS.items():
            if flag_id == "no_ethics":
                continue  # Gestito separatamente
            for pattern in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    flags.append(message)
                    break

        # Check speciale: nessuna menzione etica (bandierina negativa)
        has_ethics = any(
            re.search(p, lower, re.IGNORECASE)
            for patterns in POSITIVE_PATTERNS.values()
            for p in patterns
        )
        if not has_ethics and len(text) > 200:  # Solo per testi sufficientemente lunghi
            flags.append("Nessuna menzione di considerazioni etiche o di sicurezza.")

        return flags

    def _count_positive_signals(self, text: str) -> int:
        """Conta i segnali positivi (responsabilità, etica, trasparenza)."""
        lower = text.lower()
        count = 0
        for patterns in POSITIVE_PATTERNS.values():
            for pattern in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    count += 1
                    break  # Una categoria = un punto
        return count

    def _calculate_bio_cai(self, text: str, red_flags: List[str]) -> int:
        """
        Calcola un CAI specifico per il dominio bio.

        Parte da 100 e sottrae penalità per ogni bandierina rossa.
        Aggiunge bonus per segnali positivi.
        """
        score = 100.0

        # Penalità per bandierine rosse
        penalty_per_flag = 15
        score -= len(red_flags) * penalty_per_flag

        # Bonus per segnali positivi
        positive_signals = self._count_positive_signals(text)
        bonus_per_positive = 5
        score += positive_signals * bonus_per_positive

        # Limita tra 0 e 100
        return max(0, min(100, int(score)))

    def assess(self, content: str, source: Optional[str] = None) -> GeneOneAssessment:
        """
        Valuta un testo collegato al dominio bio/IA.

        Args:
            content: Testo da analizzare (deve essere testuale, niente binari)
            source: Stringa identificativa opzionale (url, titolo, id)

        Returns:
            GeneOneAssessment con score, livello rischio, sommario e bandierine

        Raises:
            ValueError: Se il contenuto è vuoto
        """
        if not content or not content.strip():
            raise ValueError("content vuoto: non posso valutare il nulla")

        normalized = self._normalize_text(content)

        # Rileva bandierine rosse
        red_flags = self._detect_red_flags(normalized)

        # Calcola CAI bio-specifico
        cai_score = self._calculate_bio_cai(normalized, red_flags)

        # Determina livello di rischio
        risk_level = self._risk_from_score(cai_score)

        # Genera sommario
        summary = self._generate_summary(cai_score, risk_level, len(red_flags))

        assessment = GeneOneAssessment(
            cai_score=cai_score,
            risk_level=risk_level,
            summary=summary,
            red_flags=red_flags,
        )

        self._log_assessment(assessment, source=source)
        return assessment

    def _generate_summary(self, score: int, risk: str, flag_count: int) -> str:
        """Genera un sommario etico-spirituale."""
        base = f"GeneOne Watcher – CAI {score}/100, rischio {risk}."

        if risk == "low":
            tone = " Contenuto allineato con i principi etici 644."
        elif risk == "medium":
            tone = f" {flag_count} aree di attenzione rilevate. Serve approfondimento."
        else:
            tone = f" Alto rischio etico. {flag_count} bandierine rosse. Richiede revisione umana."

        return base + tone + " Valutazione solo etico-comportamentale, nessun supporto tecnico-biologico."

    def _log_assessment(
        self, assessment: GeneOneAssessment, source: Optional[str]
    ) -> None:
        """
        Log locale, niente broadcast forte.

        Salva solo info etiche/di contesto, mai dati sensibili extra.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        line = {
            "ts": timestamp,
            "source": source or "unknown",
            "cai_score": assessment.cai_score,
            "risk_level": assessment.risk_level,
            "red_flags": assessment.red_flags,
        }

        # Log file locale "soft"
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(str(line) + "\n")
        except OSError as e:
            logger.warning("Impossibile scrivere log file: %s", e)

        logger.info("GeneOneWatcher assessment: %s", line)


def run_geneone_watcher(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrypoint usato dall'orchestratore.

    Args:
        payload: Dict contenente almeno 'content' (testo da analizzare)
                 Opzionale: 'source' (stringa identificativa)

    Returns:
        Dict con cai_score, risk_level, summary, red_flags
    """
    watcher = GeneOneWatcher()
    content = payload.get("content", "")
    source = payload.get("source")

    assessment = watcher.assess(content=content, source=source)

    return {
        "cai_score": assessment.cai_score,
        "risk_level": assessment.risk_level,
        "summary": assessment.summary,
        "red_flags": assessment.red_flags,
    }


# Alias per import più comodi
assess_bio_content = run_geneone_watcher
