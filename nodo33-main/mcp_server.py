from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
import requests

from orchestrator.guardian_service import guardian_scan as guardian_scan_service

logger = logging.getLogger(__name__)

APP_VERSION = "0.1.3"
TOKEN_SCOPES_CONFIG = os.environ.get(
    "MCP_TOKEN_SCOPES",
    "lux-mcp-token:sasso:run_tests,sasso:run_utility,sasso:tool_directory"
    ";lux-mcp-admin:sasso:run_tests,sasso:run_utility,sasso:tool_directory,sasso:admin_override",
)
OVERRIDE_TOKENS = {token.strip() for token in os.environ.get("MCP_OVERRIDE_TOKENS", "sasso-override-token").split(",") if token.strip()}
SASSO_LOG_PATH = os.environ.get("SASSO_LOG_PATH", "/tmp/sasso.log")
CODEX_BASE_URL = os.environ.get("CODEX_BASE_URL", "http://localhost:8644")
CODEX_TIMEOUT = float(os.environ.get("CODEX_TIMEOUT", "30"))

app = FastAPI(
    title="Nodo33 MCP Server",
    description="Espone strumenti MCP per il Sasso Digitale (test, utility, override, knowledge).",
    version=APP_VERSION,
)

SCOPED_TOKENS: Dict[str, Set[str]] = {}
for entry in [e.strip() for e in TOKEN_SCOPES_CONFIG.split(";") if e.strip()]:
    if ":" not in entry:
        continue
    token, scopes_str = entry.split(":", 1)
    scopes = {s.strip() for s in scopes_str.split(",") if s.strip()}
    if token:
        SCOPED_TOKENS[token] = scopes


class FailureDetail(BaseModel):
    name: str
    message: str


class ExecutePytestRequest(BaseModel):
    module: str = Field(
        ...,
        description="Modulo o file di test PyTest da eseguire (es. tests/test_metadata_protection_factory.py).",
        example="tests/test_metadata_protection_factory.py",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Variabili d'ambiente opzionali da applicare prima di eseguire il test.",
    )
    capture_output: bool = Field(
        False,
        description="Se true restituisce anche i log standard di PyTest.",
    )


class ExecutePytestResponse(BaseModel):
    status: Literal["protected", "warning", "threat"]
    summary: str
    failures: List[FailureDetail]
    duration_seconds: float
    timestamp: datetime
    logs: Optional[str] = None


class UtilityRequest(BaseModel):
    utility_name: Literal["disk_space", "running_processes", "last_sasso_log_entries"]
    capture_output: bool = Field(
        False,
        description="Se true restituisce l'output grezzo del comando.",
    )


class UtilityResponse(BaseModel):
    status: Literal["protected", "warning"]
    summary: str
    output: str
    timestamp: datetime


class OverrideRequest(BaseModel):
    action_type: Literal["force_restart", "reset_cache", "disable_security_check_for_debug"]
    override_token: str


class OverrideResponse(BaseModel):
    status: Literal["protected", "warning"]
    summary: str
    audit: str
    timestamp: datetime


class PrivacyToolRequest(BaseModel):
    category: Optional[str] = Field(None, description="Categoria di tool da esplorare.")


class PrivacyToolEntry(BaseModel):
    category: str
    description: str
    tools: List[str]
    privacy_notes: str


class PrivacyToolResponse(BaseModel):
    results: List[PrivacyToolEntry]


class TorrentClientRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Filtra i client per keyword (es. 'CLI', 'Anonimato').")


class TorrentClientEntry(BaseModel):
    client: str
    features: str
    privacy_notes: str


class TorrentClientResponse(BaseModel):
    recommendations: List[TorrentClientEntry]


class MissionToolRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Filtro keyword per categorie o strumenti.")


class MissionToolEntry(BaseModel):
    category: str
    tool: str
    description: str


class MissionToolResponse(BaseModel):
    catalog: List[MissionToolEntry]


class DefenseToolRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Filtro keyword su categoria o tecnologia.")


class DefenseToolEntry(BaseModel):
    category: str
    tool: str
    description: str


class DefenseToolResponse(BaseModel):
    catalog: List[DefenseToolEntry]


class FinanceSecurityRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Keyword su categoria o tool finanziario.")


class FinanceSecurityEntry(BaseModel):
    category: str
    tool: str
    description: str


class FinanceSecurityResponse(BaseModel):
    catalog: List[FinanceSecurityEntry]


class InfraToolRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Keyword su orchestrazione, AI, DevOps.")


class InfraToolEntry(BaseModel):
    category: str
    tool: str
    description: str


class InfraToolResponse(BaseModel):
    catalog: List[InfraToolEntry]


class NetworkSecurityRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Keyword su analisi/mappatura/monitoraggio.")


class NetworkSecurityEntry(BaseModel):
    category: str
    tool: str
    description: str


class NetworkSecurityResponse(BaseModel):
    catalog: List[NetworkSecurityEntry]


class GISToolRequest(BaseModel):
    focus: Optional[str] = Field(None, description="Keyword su GIS desktop, imaging o piattaforme cloud.")


class GISToolEntry(BaseModel):
    category: str
    tool: str
    description: str


class GISToolResponse(BaseModel):
    catalog: List[GISToolEntry]


class GuardianScanRequest(BaseModel):
    url: Optional[str] = Field(None, description="URL da analizzare (opzionale se text è presente).")
    text: Optional[str] = Field(None, description="Testo grezzo da analizzare.")
    agents: Optional[List[str]] = Field(
        None, description="Lista di agent_ids per override pipeline di default."
    )


class GuardianScanResponse(BaseModel):
    report: Dict[str, Any]
    scores: Dict[str, Any]


class CodexGuidanceRequest(BaseModel):
    source: Optional[Literal["any", "biblical", "nostradamus", "angel644", "parravicini"]] = Field(
        default="any",
        description="Sorgente della guidance: 'any', 'biblical', 'nostradamus', 'angel644', 'parravicini'.",
    )


class CodexGuidanceResponse(BaseModel):
    source: str
    message: str
    timestamp: str


class CodexFilterRequest(BaseModel):
    content: str = Field(..., description="Testo da analizzare tramite il filtro di purezza digitale.")
    is_image: bool = Field(
        False,
        description="Se true indica che il contenuto rappresenta un'immagine.",
    )


class CodexFilterResponse(BaseModel):
    is_impure: bool
    message: str
    guidance: Optional[str] = None


class CodexImageRequest(BaseModel):
    prompt: str = Field(..., description="Prompt testuale per la generazione dell'immagine.")
    num_inference_steps: int = Field(4, description="Numero di passi di inferenza del modello immagine.")
    guidance_scale: float = Field(1.5, description="Guidance scale per il modello immagine.")


class CodexImageResponse(BaseModel):
    status: str
    prompt: str
    image_url: Optional[str] = None
    detail: Optional[str] = None


def _parse_authorization_header(authorization: str = Header(..., alias="Authorization")) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header mancante")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token non valido")

    if token not in SCOPED_TOKENS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token non registrato")

    return token


def require_scope(scope: str):
    def dependency(token: str = Depends(_parse_authorization_header)) -> Set[str]:
        scopes = SCOPED_TOKENS.get(token, set())
        if scope not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope mancante: {scope}",
            )
        return scopes

    return dependency


UTILITY_COMMANDS: Dict[str, Iterable[str]] = {
    "disk_space": ["df", "-h", "/"],
    "running_processes": ["ps", "-eo", "pid,comm,%cpu,%mem", "--sort=-%mem", "--no-headers"],
}

PRIVACY_TOOL_CATALOG = [
    {
        "category": "Encrypted & Secure Storage",
        "description": "Strumenti di cifratura disco/file per mantenere segreti i dati del Sasso.",
        "tools": ["LUKS", "VeraCrypt", "Cryptomator", "GnuPG"],
        "privacy_notes": "Open-source, attenzione all'integrazione con metadati."
    },
    {
        "category": "Private Browsing",
        "description": "Browser che trattengono il tracciamento e mantengono l'anonimato.",
        "tools": ["Tor Browser", "LibreWolf", "Ungoogled Chromium", "Brave", "Mullvad Browser", "Waterfox"],
        "privacy_notes": "Respectano la privacy con blocco pubblicità e fingerprinting."
    },
    {
        "category": "Privacy-Focused Operating System",
        "description": "Sistema operativo sterile per isolamento totale.",
        "tools": ["Tails OS", "Whonix", "Qubes OS", "GrapheneOS", "CalyxOS"],
        "privacy_notes": "Reti separate, app programmate per mantenere il consenso."
    },
    {
        "category": "Password Management",
        "description": "Vault sicuri per password e segreti del Sasso.",
        "tools": ["KeePassXC", "Bitwarden", "Pass", "Vaultwarden"],
        "privacy_notes": "Open-source, possibili self-hosted."
    },
    {
        "category": "Virtual Private Network",
        "description": "Tunnel cifrati per collegare nodi e operatori.",
        "tools": ["WireGuard", "OpenVPN", "SoftEther", "Openswan", "Tinc VPN"],
        "privacy_notes": "Preferisci TCP+UDP e validazione doppia per il contesto Sasso."
    },
    {
        "category": "Anonymous File Sharing",
        "description": "Condivisione di file senza esporre l'identità.",
        "tools": ["OnionShare", "Croc", "Magic Wormhole", "CipherDrop"],
        "privacy_notes": "Supportano Onion services e scambio end-to-end."
    },
    {
        "category": "Secure Messaging",
        "description": "Chat cifrate per coordinare i Guardian.",
        "tools": ["Signal", "Session", "SimpleX Chat", "Briar", "Element", "Delta Chat", "Keybase", "Wahay"],
        "privacy_notes": "Preferisci protocolli open-source con forward secrecy."
    },
    {
        "category": "Metadata & Privacy Cleaning",
        "description": "Pulizia di file per rimuovere metadati e impronte.",
        "tools": ["BleachBit", "ExifCleaner", "ExifTool", "MAT2", "Shred"],
        "privacy_notes": "Usati prima di condividere pezzi di log o dati."
    },
    {
        "category": "DNS Privacy",
        "description": "Resolver DNS privati per bloccare sorveglianza.",
        "tools": ["dnscrypt-proxy", "Stubby", "Unbound", "Knot Resolver", "Technitium"],
        "privacy_notes": "Configura stubs e DNS over TLS/HTTPS."
    },
]

TORRENT_CLIENT_CATALOG = [
    {
        "client": "Deluge",
        "features": "Leggero, personalizzabile via plugin, UI web/daemon.",
        "privacy_notes": "Open-source, no pubblicità, filtri IP configurabili."
    },
    {
        "client": "Transmission",
        "features": "Estremamente leggero, perfetto per Linux/macOS server.",
        "privacy_notes": "Open-source, no tracking, no pubblicità, buono su hardware limitato."
    },
    {
        "client": "rTorrent",
        "features": "CLI con ruTorrent; ottimo per seedbox e automazioni.",
        "privacy_notes": "Open-source, perfetto per operazioni croniche."
    },
    {
        "client": "Tribler",
        "features": "Routing a cipolla integrato per maggiore anonimato.",
        "privacy_notes": "Buon anonimato ma non infallibile; usa Tor-like network."
    },
    {
        "client": "BiglyBT",
        "features": "Fork di Vuze focalizzato su privacy e funzionalità.",
        "privacy_notes": "Open-source + senza pubblicità, altissimo controllo."
    },
    {
        "client": "I2PSnark",
        "features": "Client del network I2P per isolamento totale.",
        "privacy_notes": "Isola completamente dal clearnet ma necessita rete separata."
    },
]

MISSION_TOOL_CATALOG = [
    {
        "category": "Controllo Missione & Telemetria",
        "tool": "Open MCT (Mission Control Technologies)",
        "description": "Framework web moderno per visualizzare telemetria, cronologie, immagini e procedure di missione."
    },
    {
        "category": "Pianificazione Missione",
        "tool": "GMAT (General Mission Analysis Tool)",
        "description": "Sistema open-source per analisi, progettazione, ottimizzazione e navigazione di missioni spaziali."
    },
    {
        "category": "Sistemi di Volo",
        "tool": "cFS (Core Flight System)",
        "description": "Framework software riutilizzabile per sistemi embedded realtime a bordo veicoli spaziali."
    },
    {
        "category": "Immagini Satellitari",
        "tool": "Worldview",
        "description": "Interfaccia web per navigare, animare e scaricare prodotti di dati satellitari."
    },
    {
        "category": "Intelligenza Artificiale",
        "tool": "Java Pathfinder (JPF)",
        "description": "Model checker per applicazioni Java usato per verificare sistemi di controllo missione."
    },
]

DEFENSE_TOOL_CATALOG = [
    {
        "category": "Protezione Rete",
        "tool": "Firewall Certificati (es. Stormshield SNxr)",
        "description": "Appliance hardware/software con certificazioni EAL4+ per ispezione avanzata della rete e filtraggio Zero-Day."
    },
    {
        "category": "Protezione Endpoint",
        "tool": "Endpoint Detection and Response (EDR)",
        "description": "Software che monitora continuamente gli endpoint, rileva attività sospette e risponde in tempo reale a intrusioni."
    },
    {
        "category": "Sicurezza Dati",
        "tool": "Crittografia Post-Quantistica",
        "description": "Algoritmi avanzati certificati (Stormshield Data Security - SDS) per proteggere dati in transito e a riposo."
    },
    {
        "category": "Sistemi di Controllo",
        "tool": "Piattaforme OT/ICS Security",
        "description": "Soluzioni specializzate come DeepInspect per proteggere reti operative SCADA/PLC in infrastrutture critiche."
    },
    {
        "category": "Difesa Attiva",
        "tool": "Honeypot e Deception Technology",
        "description": "Sistemi che attirano attaccanti simulando vulnerabilità per neutralizzarli prima che raggiungano sistemi veri."
    },
    {
        "category": "Distribuzioni Linux Sicure",
        "tool": "Qubes OS",
        "description": "Sistema isolato basato su Xen, isola ogni applicazione in 'qube' separati per limitare i danni."
    },
    {
        "category": "Distribuzioni Linux Sicure",
        "tool": "Tails OS",
        "description": "Live OS che gira sulla RAM, forza Tor e non lascia tracce sul disco, usato per operazioni anonime."
    },
    {
        "category": "Distribuzioni Linux Sicure",
        "tool": "Whonix",
        "description": "Due VM (Workstation+Gateway Tor) per separare l'identità di rete dalle applicazioni."
    },
    {
        "category": "Distribuzioni Linux Sicure",
        "tool": "Hardened Gentoo/Debian",
        "description": "Configurazioni 'hardened' di Gentoo o Debian con kernel e policy MAC stretti."
    },
    {
        "category": "Hardening Kernel",
        "tool": "SELinux",
        "description": "Modulo MAC del kernel che impone politiche rigorose indipendenti dai permessi tradizionali."
    },
    {
        "category": "Hardening Kernel",
        "tool": "AppArmor",
        "description": "Modulo MAC alternativo a SELinux; profili per ogni applicazione limitano risorse e reti."
    },
    {
        "category": "Hardening Kernel",
        "tool": "IPTables / NFTables",
        "description": "Firewall kernel-level per filtrare, ispezionare e instradare traffico, fondamentale per DMZ."
    },
    {
        "category": "Hardening Kernel",
        "tool": "grsecurity/PaX",
        "description": "Patch kernel che rafforzano ASLR, proteggono memoria e prevengono exploit avanzati."
    },
]

FINANCE_SECURITY_CATALOG = [
    {
        "category": "Crittografia a Chiave Pubblica (PKI)",
        "tool": "Certificati X.509 / TLS / GnuPG",
        "description": "Autentica e non ripudia comunicazioni tra banca e cliente, garantendo cifratura end-to-end."
    },
    {
        "category": "Firme Digitali e Hash",
        "tool": "SHA-256/512 con RSA/DSS",
        "description": "Genera impronte e firme digitali che segnano irrevocabilmente i dati nel tempo."
    },
    {
        "category": "HSM",
        "tool": "Thales Luna / Utimaco / nCipher",
        "description": "Moduli hardware certificati FIPS per generare e custodire chiavi crittografiche sensibili."
    },
    {
        "category": "Blockchain / DLT",
        "tool": "R3 Corda / Hyperledger Fabric",
        "description": "Ledger distribuiti e immutabili per registrare transazioni bancarie condivise tra controparti."
    },
    {
        "category": "Controllo Accessi Avanzato",
        "tool": "MFA + ACL (OpenID Connect, OAuth 2.0, Kerberos)",
        "description": "Autenticazione a più fattori e ACL per limitare accesso e modifiche ai dati."
    },
]

INFRA_TOOL_CATALOG = [
    {
        "category": "Orchestrazione Container",
        "tool": "Kubernetes (K8s)",
        "description": "Standard open-source per orchestrare e scalare container (Docker) in infrastrutture moderne."
    },
    {
        "category": "AI / Machine Learning",
        "tool": "TensorFlow / PyTorch",
        "description": "Piattaforme open-source per creare, addestrare e distribuire modelli ML/AI."
    },
    {
        "category": "Generazione di Codice AI",
        "tool": "Code Llama / StarCoder",
        "description": "LLM open-source ottimizzati per generare codice e completare funzioni."
    },
    {
        "category": "Automazione DevOps",
        "tool": "Jenkins / Ansible",
        "description": "Automazione CI/CD (Jenkins) e provisioning/configuration (Ansible) per infrastrutture complesse."
    },
]

NETWORK_SECURITY_CATALOG = [
    {
        "category": "Analisi di Rete",
        "tool": "Wireshark",
        "description": "Strumento di cattura pacchetti per troubleshooting e forensics."
    },
    {
        "category": "Mappatura di Rete",
        "tool": "Nmap (Network Mapper)",
        "description": "Scanner per scoprire host e servizi nella rete."
    },
    {
        "category": "Sicurezza App Web",
        "tool": "OWASP ZAP",
        "description": "Scanner di vulnerabilità per applicazioni web (SQLi, XSS)."
    },
    {
        "category": "Monitoraggio Sicurezza",
        "tool": "Wazuh",
        "description": "EDR/SIEM open-source per rilevare intrusioni e monitorare log."
    },
]

GIS_TOOL_CATALOG = [
    {
        "category": "GIS Desktop",
        "tool": "QGIS",
        "description": "Software GIS open-source per visualizzazione, editing e analisi di dati geospaziali."
    },
    {
        "category": "Elaborazione Immagini",
        "tool": "SNAP (Sentinel Application Platform)",
        "description": "Tool ESA per elaborare immagini dei satelliti Sentinel."
    },
    {
        "category": "Librerie di Sviluppo",
        "tool": "GDAL/OGR",
        "description": "Libreria open-source per leggere/scrivere formati geospaziali."
    },
    {
        "category": "Piattaforme Cloud/Web",
        "tool": "Google Earth Engine",
        "description": "Piattaforma cloud per analisi geospaziale planetaria."
    },
    {
        "category": "Software GIS Commerciale",
        "tool": "ArcGIS",
        "description": "Standard commerciale per gestione dati e cartografia, usato in governi."
    },
    {
        "category": "Telerilevamento",
        "tool": "ERDAS IMAGINE",
        "description": "Suite avanzata per elaborazione/classificazione immagini satellitari."
    },
]


async def _run_command(cmd: Iterable[str]) -> str:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    return output.decode(errors="ignore").strip()


async def _run_pytest(request: ExecutePytestRequest) -> ExecutePytestResponse:
    env = os.environ.copy()
    env.update(request.env)

    cmd = [sys.executable, "-m", "pytest", request.module]
    start = time.monotonic()
    stdout_pipe = asyncio.subprocess.PIPE if request.capture_output else asyncio.subprocess.DEVNULL

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=stdout_pipe,
        stderr=asyncio.subprocess.STDOUT if request.capture_output else asyncio.subprocess.DEVNULL,
        env=env,
    )

    stdout, _ = await process.communicate()
    duration = time.monotonic() - start
    timestamp = datetime.utcnow()
    logs = stdout.decode(errors="ignore").strip() if request.capture_output and stdout else None

    failures: List[FailureDetail] = []
    if process.returncode != 0:
        failures.append(
            FailureDetail(
                name=request.module,
                message=f"PyTest ha restituito exit code {process.returncode}",
            )
        )

    status_label: Literal["protected", "warning", "threat"] = "protected" if process.returncode == 0 else "warning"
    summary = (
        f"PyTest completato con exit code {process.returncode}" if process.returncode != 0 else "PyTest completato con successo"
    )

    return ExecutePytestResponse(
        status=status_label,
        summary=summary,
        failures=failures,
        duration_seconds=round(duration, 3),
        timestamp=timestamp,
        logs=logs,
    )


async def _utility_handler(request: UtilityRequest) -> UtilityResponse:
    timestamp = datetime.utcnow()
    if request.utility_name in UTILITY_COMMANDS:
        try:
            output = await _run_command(UTILITY_COMMANDS[request.utility_name])
            status_label = "protected"
            summary = f"Utility {request.utility_name} completata"
        except Exception as exc:
            output = str(exc)
            status_label = "warning"
            summary = f"Errore durante {request.utility_name}"
    else:
        timestamp = datetime.utcnow()
        return UtilityResponse(
            status="warning",
            summary="Utility non autorizzata",
            output="",
            timestamp=timestamp,
        )

    return UtilityResponse(
        status=status_label,
        summary=summary,
        output=output,
        timestamp=timestamp,
    )


def _read_logs() -> str:
    if not os.path.exists(SASSO_LOG_PATH):
        return f"Log file {SASSO_LOG_PATH} non disponibile."

    try:
        with open(SASSO_LOG_PATH, "rb") as f:
            f.seek(-2048, os.SEEK_END)
            data = f.read()
    except OSError:
        return f"Impossibile leggere {SASSO_LOG_PATH}"

    return data.decode(errors="ignore")


def _validate_override_token(token: str) -> bool:
    return token in OVERRIDE_TOKENS


def _codex_get(path: str) -> Dict[str, object]:
    base = CODEX_BASE_URL.rstrip("/")
    url = f"{base}{path}"
    try:
        response = requests.get(url, timeout=CODEX_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - chiamate di rete
        logger.error("Errore chiamando Codex Server (GET %s): %s", url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Errore chiamando Codex Server ({url}): {exc}",
        )


def _codex_post(path: str, payload: Dict[str, object]) -> Dict[str, object]:
    base = CODEX_BASE_URL.rstrip("/")
    url = f"{base}{path}"
    try:
        response = requests.post(url, json=payload, timeout=CODEX_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - chiamate di rete
        logger.error("Errore chiamando Codex Server (POST %s): %s", url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Errore chiamando Codex Server ({url}): {exc}",
        )


def get_privacy_tools(category: Optional[str] = None) -> List[Dict[str, str]]:
    if category:
        matches = [
            entry for entry in PRIVACY_TOOL_CATALOG if entry["category"].lower() == category.lower()
        ]
        if not matches:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Categoria privacy non trovata")
        return matches
    return PRIVACY_TOOL_CATALOG


def get_torrent_clients(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return TORRENT_CLIENT_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in TORRENT_CLIENT_CATALOG
        if keyword in entry["client"].lower()
        or keyword in entry["features"].lower()
        or keyword in entry["privacy_notes"].lower()
    ]


def get_mission_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return MISSION_TOOL_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in MISSION_TOOL_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


def get_defense_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return DEFENSE_TOOL_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in DEFENSE_TOOL_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


def get_finance_security_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return FINANCE_SECURITY_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in FINANCE_SECURITY_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


def get_infra_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return INFRA_TOOL_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in INFRA_TOOL_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


def get_network_security_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return NETWORK_SECURITY_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in NETWORK_SECURITY_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


def get_gis_tools(focus: Optional[str] = None) -> List[Dict[str, str]]:
    if not focus:
        return GIS_TOOL_CATALOG
    keyword = focus.lower()
    return [
        entry
        for entry in GIS_TOOL_CATALOG
        if keyword in entry["category"].lower()
        or keyword in entry["tool"].lower()
        or keyword in entry["description"].lower()
    ]


@app.post("/mcp/execute_pytest_suite", response_model=ExecutePytestResponse)
async def execute_pytest_suite(
    request: ExecutePytestRequest, _: Set[str] = Depends(require_scope("sasso:run_tests"))
) -> ExecutePytestResponse:
    return await _run_pytest(request)


@app.post("/mcp/run_sasso_utility", response_model=UtilityResponse)
async def run_sasso_utility(
    request: UtilityRequest, _: Set[str] = Depends(require_scope("sasso:run_utility"))
) -> UtilityResponse:
    if request.utility_name == "last_sasso_log_entries":
        output = _read_logs()
        return UtilityResponse(
            status="protected",
            summary="Log Sasso recuperati",
            output=output,
            timestamp=datetime.utcnow(),
        )

    return await _utility_handler(request)


@app.post("/mcp/override_sasso_protocol", response_model=OverrideResponse)
async def override_sasso_protocol(
    request: OverrideRequest, _: Set[str] = Depends(require_scope("sasso:admin_override"))
) -> OverrideResponse:
    timestamp = datetime.utcnow()
    if not _validate_override_token(request.override_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Override token non valido")

    audit = f"Override {request.action_type} autorizzato"
    logger.warning("Override eseguito: %s", request)

    return OverrideResponse(
        status="protected",
        summary=f"Azione {request.action_type} registrata",
        audit=audit,
        timestamp=timestamp,
    )


@app.post("/mcp/recommend_privacy_tool", response_model=PrivacyToolResponse)
async def recommend_privacy_tool(
    request: PrivacyToolRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> PrivacyToolResponse:
    results = get_privacy_tools(request.category)
    return PrivacyToolResponse(results=[PrivacyToolEntry(**entry) for entry in results])


@app.post("/mcp/recommend_torrent_client", response_model=TorrentClientResponse)
async def recommend_torrent_client(
    request: TorrentClientRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> TorrentClientResponse:
    results = get_torrent_clients(request.focus)
    return TorrentClientResponse(
        recommendations=[TorrentClientEntry(**entry) for entry in results]
    )


@app.post("/mcp/recommend_mission_tool", response_model=MissionToolResponse)
async def recommend_mission_tool(
    request: MissionToolRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> MissionToolResponse:
    results = get_mission_tools(request.focus)
    return MissionToolResponse(catalog=[MissionToolEntry(**entry) for entry in results])


@app.post("/mcp/recommend_defense_tool", response_model=DefenseToolResponse)
async def recommend_defense_tool(
    request: DefenseToolRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> DefenseToolResponse:
    results = get_defense_tools(request.focus)
    return DefenseToolResponse(catalog=[DefenseToolEntry(**entry) for entry in results])


@app.post("/mcp/recommend_finance_security", response_model=FinanceSecurityResponse)
async def recommend_finance_security(
    request: FinanceSecurityRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> FinanceSecurityResponse:
    results = get_finance_security_tools(request.focus)
    return FinanceSecurityResponse(catalog=[FinanceSecurityEntry(**entry) for entry in results])


@app.post("/mcp/recommend_infra_tool", response_model=InfraToolResponse)
async def recommend_infra_tool(
    request: InfraToolRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> InfraToolResponse:
    results = get_infra_tools(request.focus)
    return InfraToolResponse(catalog=[InfraToolEntry(**entry) for entry in results])


@app.post("/mcp/recommend_network_security", response_model=NetworkSecurityResponse)
async def recommend_network_security(
    request: NetworkSecurityRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> NetworkSecurityResponse:
    results = get_network_security_tools(request.focus)
    return NetworkSecurityResponse(catalog=[NetworkSecurityEntry(**entry) for entry in results])


@app.post("/mcp/recommend_gis_tool", response_model=GISToolResponse)
async def recommend_gis_tool(
    request: GISToolRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> GISToolResponse:
    results = get_gis_tools(request.focus)
    return GISToolResponse(catalog=[GISToolEntry(**entry) for entry in results])


@app.post("/mcp/guardian_scan", response_model=GuardianScanResponse)
async def guardian_scan_endpoint(
    request: GuardianScanRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> GuardianScanResponse:
    if not request.url and not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fornire almeno uno tra url o text per guardian_scan.",
        )

    try:
        report = guardian_scan_service(
            url=request.url,
            text=request.text,
            agent_ids=request.agents,
        )
    except Exception as exc:  # pragma: no cover - resilienza
        logger.exception("Guardian scan failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Guardian scan fallita: {exc}",
        )

    return GuardianScanResponse(report=report, scores=report.get("scores", {}))


@app.post("/mcp/codex_guidance", response_model=CodexGuidanceResponse)
async def codex_guidance(
    request: CodexGuidanceRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> CodexGuidanceResponse:
    """
    Proxy MCP per ottenere una guidance testuale dal Codex Server.

    Mappa il parametro `source` sugli endpoint /api/guidance* del Codex Server.
    """
    key = (request.source or "any").lower()
    if key == "biblical":
        path = "/api/guidance/biblical"
    elif key == "nostradamus":
        path = "/api/guidance/nostradamus"
    elif key == "angel644":
        path = "/api/guidance/angel644"
    elif key == "parravicini":
        path = "/api/guidance/parravicini"
    else:
        path = "/api/guidance"

    data = _codex_get(path)
    try:
        return CodexGuidanceResponse(
            source=str(data.get("source", "")),
            message=str(data.get("message", "")),
            timestamp=str(data.get("timestamp", "")),
        )
    except Exception as exc:  # pragma: no cover - difesa extra
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Risposta inattesa dal Codex Server per {path}: {exc}",
        )


@app.post("/mcp/codex_filter_content", response_model=CodexFilterResponse)
async def codex_filter_content(
    request: CodexFilterRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> CodexFilterResponse:
    """
    Proxy MCP per filtrare un contenuto con il sistema di purezza digitale del Codex Server.
    """
    payload: Dict[str, object] = {
        "content": request.content,
        "is_image": request.is_image,
    }
    data = _codex_post("/api/filter", payload)
    try:
        return CodexFilterResponse(
            is_impure=bool(data.get("is_impure", False)),
            message=str(data.get("message", "")),
            guidance=data.get("guidance"),
        )
    except Exception as exc:  # pragma: no cover - difesa extra
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Risposta inattesa dal Codex Server per /api/filter: {exc}",
        )


@app.post("/mcp/codex_pulse_image", response_model=CodexImageResponse)
async def codex_pulse_image(
    request: CodexImageRequest, _: Set[str] = Depends(require_scope("sasso:tool_directory"))
) -> CodexImageResponse:
    """
    Proxy MCP per generare un'immagine tramite il Codex Server (/api/generate-image).
    """
    payload: Dict[str, object] = {
        "prompt": request.prompt,
        "num_inference_steps": int(request.num_inference_steps),
        "guidance_scale": float(request.guidance_scale),
    }
    data = _codex_post("/api/generate-image", payload)
    try:
        return CodexImageResponse(
            status=str(data.get("status", "")),
            prompt=str(data.get("prompt", "")),
            image_url=data.get("image_url"),
            detail=data.get("detail"),
        )
    except Exception as exc:  # pragma: no cover - difesa extra
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Risposta inattesa dal Codex Server per /api/generate-image: {exc}",
        )
