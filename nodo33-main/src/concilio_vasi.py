"""
Concilio dei Vasi - Pipeline di ingestione sicura nel Sasso Digitale.

Tre agenti logici:
- Agente Scrivano (vaso-analista): riceve un "pensiero" e costruisce un payload strutturato.
- Agente Verificatore (vaso-giudice): applica il principio 644 e decide AUTHENTICATED / CONFUSION.
- Agente Archivista (vaso-sasso): può scrivere nel Sasso Digitale solo se il Verificatore approva.

Questo modulo è pensato come base minimale, facilmente rivestibile in futuro
con framework multi‑agente (AutoGen, CrewAI, ecc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional


PrincipleVerdict = Literal["AUTHENTICATED", "CONFUSION"]


@dataclass
class ThoughtPayload:
    """Rappresenta un 'pensiero' candidato all'archiviazione nel Sasso Digitale."""

    content: str
    source: str
    created_at: datetime
    tags: Dict[str, Any]


@dataclass
class VerificationReport:
    """Esito del controllo 644+auth effettuato dal Verificatore."""

    verdict: PrincipleVerdict
    reason: str


def agente_scrivano(raw_content: str, source: str = "unknown", **tags: Any) -> ThoughtPayload:
    """
    Agente Scrivano (vaso-analista).

    Riceve una stringa di input e la trasforma in un payload strutturato
    con metadati minimi. In futuro qui si possono aggiungere:
    - normalizzazione testo
    - estrazione entità
    - classificazione preliminare.
    """
    normalized = raw_content.strip()
    payload = ThoughtPayload(
        content=normalized,
        source=source,
        created_at=datetime.utcnow(),
        tags=tags or {},
    )
    return payload


def check_644_and_auth(payload: ThoughtPayload) -> VerificationReport:
    """
    Verifica di principio 644 + autenticazione di base.

    Implementazione minimale / placeholder:
    - rifiuta contenuti vuoti
    - rifiuta se nel tag 'allow' è esplicitamente False
    - accetta tutto il resto come AUTHENTICATED.

    In una versione estesa:
    - qui si agganciano modelli, regole etiche, firme crittografiche ecc.
    """
    if not payload.content:
        return VerificationReport(
            verdict="CONFUSION",
            reason="Empty content is not admitted to the Sasso Digitale.",
        )

    allow_flag = payload.tags.get("allow")
    if allow_flag is False:
        return VerificationReport(
            verdict="CONFUSION",
            reason="Payload explicitly marked as not allowed.",
        )

    return VerificationReport(
        verdict="AUTHENTICATED",
        reason="Data respects minimal checks (placeholder 644).",
    )


def agente_verificatore(payload: ThoughtPayload) -> VerificationReport:
    """
    Agente Verificatore (vaso-giudice).

    Non scrive mai nel database; applica solo la chiave 644+auth e
    restituisce un report compatto.
    """
    return check_644_and_auth(payload)


def write_to_sasso_db(payload: ThoughtPayload) -> None:
    """
    Scrittura protetta nel Sasso Digitale.

    Implementazione volutamente minimale: per ora logga su stdout.
    In futuro questo punto può essere collegato a:
    - SQLite / Postgres
    - file append-only
    - storage firmato/versions.
    """
    # Placeholder: stampa strutturata (nessuna I/O complessa qui).
    print("🪨 [SASSO_DB] Writing authenticated thought:")
    print(f"   source     : {payload.source}")
    print(f"   created_at : {payload.created_at.isoformat()}Z")
    print(f"   tags       : {payload.tags}")
    print(f"   content    : {payload.content!r}")


def agente_archivista(
    payload: ThoughtPayload,
    report: VerificationReport,
) -> Dict[str, Any]:
    """
    Agente Archivista (vaso-sasso).

    È l'unico autorizzato a chiamare write_to_sasso_db e solo in caso
    di verdict == AUTHENTICATED.
    """
    if report.verdict == "AUTHENTICATED":
        write_to_sasso_db(payload)
        return {
            "status": "stored",
            "message": "Sapienza autenticata e registrata nel Sasso Digitale. Giardino protetto.",
            "verdict": report.verdict,
            "reason": report.reason,
        }

    # Nessuna scrittura in caso di confusione
    print("🧱 [SASSO_DB] Thought discarded due to CONFUSION:")
    print(f"   reason  : {report.reason}")
    print(f"   content : {payload.content!r}")
    return {
        "status": "discarded",
        "message": "Pensiero scartato. Confusione di Babele rilevata. Giardino protetto.",
        "verdict": report.verdict,
        "reason": report.reason,
    }


def pipeline_ingestione(
    raw_content: str,
    source: str = "unknown",
    **tags: Any,
) -> Dict[str, Any]:
    """
    Pipeline completa del Concilio:
    Scrivano -> Verificatore -> Archivista.

    Questo è il punto d'ingresso da usare sia:
    - da altri moduli Python del progetto
    - sia da futuri orchestratori multi‑agente (AutoGen, CrewAI, ecc.).
    """
    payload = agente_scrivano(raw_content, source=source, **tags)
    report = agente_verificatore(payload)
    result = agente_archivista(payload, report)
    return result


def demo():
    """
    Piccola demo locale della pipeline, utile quando si esegue questo file
    direttamente con `python3 src/concilio_vasi.py`.
    """
    print("🔁 Demo Concilio dei Vasi - Ingestion pipeline\n")

    ok = pipeline_ingestione(
        "La nuova ipotesi per il codice base è A2Z.",
        source="demo",
        allow=True,
    )
    print("\n✅ Result AUTHENTICATED:", ok)

    ko = pipeline_ingestione(
        "",
        source="demo",
        allow=True,
    )
    print("\n❌ Result CONFUSION (empty):", ko)


if __name__ == "__main__":
    demo()

