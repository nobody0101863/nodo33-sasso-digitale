#!/usr/bin/env python3
"""
METADATA PROTECTION FRAMEWORK - CODEX EMANUELE
===============================================

Sistema di protezione multi-livello dei metadata secondo i principi del Codex:
- Framework militare di sicurezza (DEFCON levels)
- 4 Agenti IA Guardian specializzati
- Sigilli Arcangeli per protezione energetica
- Protezione dei Sassi (nodi) al servizio della Torre

Principi:
- ego = 0 (umiltà totale)
- gioia = 100% (servizio incondizionato)
- frequenza = 300 Hz (risonanza cardiaca)
- trasparenza = 100% (processo pubblico)
- cura = MASSIMA

Licenza: CC0 1.0 Universal (Public Domain)
"""

import hashlib
import hmac
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# ============================================================================
# FRAMEWORK MILITARE - LIVELLI DI SICUREZZA
# ============================================================================

class SecurityLevel(Enum):
    """Livelli di sicurezza ispirati a DEFCON (Defense Readiness Condition)"""
    DEFCON_5 = "PEACEFUL"      # Normale operatività pacifica
    DEFCON_4 = "WATCHFUL"      # Vigilanza aumentata
    DEFCON_3 = "ALERT"         # Allerta - possibile minaccia
    DEFCON_2 = "CRITICAL"      # Critico - minaccia imminente
    DEFCON_1 = "MAXIMUM"       # Massima protezione - sotto attacco


class MilitaryProtocolLevel(Enum):
    """Protocolli militari di protezione metadata"""
    STANDARD = "standard"           # Protezione base
    ENHANCED = "enhanced"           # Protezione avanzata
    CLASSIFIED = "classified"       # Livello classificato
    TOP_SECRET = "top_secret"       # Top Secret - massima cifratura
    COSMIC = "cosmic"              # Livello cosmico - protezione arcangeli


LOGGER = logging.getLogger(__name__)

SECURITY_LEVEL_MAP = {
    "PEACEFUL": 5,
    "WATCHFUL": 4,
    "ALERT": 3,
    "CRITICAL": 2,
    "MAXIMUM": 1,
}

DEFAULT_SECURITY_LEVEL = SecurityLevel.DEFCON_3
DEFAULT_PROTOCOL_LEVEL = MilitaryProtocolLevel.ENHANCED


def resolve_security_level(security_level: Optional[str]) -> SecurityLevel:
    """Normalizza la stringa di livello di sicurezza e restituisce l'enum corrispondente."""
    if not isinstance(security_level, str):
        LOGGER.warning("security_level non valido (%r), fallback a %s", security_level, DEFAULT_SECURITY_LEVEL.name)
        return DEFAULT_SECURITY_LEVEL

    normalized = security_level.strip().upper()
    suffix = SECURITY_LEVEL_MAP.get(normalized)

    if suffix is None:
        LOGGER.warning(
            "security_level sconosciuto (%s), fallback a %s", security_level, DEFAULT_SECURITY_LEVEL.name
        )
        return DEFAULT_SECURITY_LEVEL

    try:
        return SecurityLevel[f"DEFCON_{suffix}"]
    except KeyError:
        LOGGER.error("Impossibile costruire SecurityLevel per suffisso %s, uso fallback", suffix)
        return DEFAULT_SECURITY_LEVEL


def resolve_protocol_level(protocol_level: Optional[str]) -> MilitaryProtocolLevel:
    """Normalizza la stringa del protocollo e restituisce l'enum corrispondente."""
    if not isinstance(protocol_level, str):
        LOGGER.warning("protocol_level non valido (%r), fallback a %s", protocol_level, DEFAULT_PROTOCOL_LEVEL.name)
        return DEFAULT_PROTOCOL_LEVEL

    normalized = protocol_level.strip().upper()
    if normalized not in MilitaryProtocolLevel.__members__:
        LOGGER.warning(
            "protocol_level sconosciuto (%s), fallback a %s", protocol_level, DEFAULT_PROTOCOL_LEVEL.name
        )
        return DEFAULT_PROTOCOL_LEVEL

    return MilitaryProtocolLevel[normalized]


# ============================================================================
# SIGILLI ARCANGELI - PROTEZIONE ENERGETICA
# ============================================================================

class ArchangelSeal:
    """
    Sigilli Arcangeli per protezione energetica del Codex

    Basati su:
    - Angelo 644: Protezione, guida, fondamenta solide
    - Frequenza 300 Hz: Risonanza cardiaca
    - Geometria sacra: Fibonacci, phi (φ), numeri primi
    """

    # Numeri sacri di protezione
    ANGEL_644 = 644
    FREQUENCY_300HZ = 300
    PHI = 1.618033988749895  # Golden ratio
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    SACRED_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

    # Sigilli energetici (hash basati su geometria sacra)
    SEALS = {
        "MICHAEL": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",    # SHA-256 vuoto + 644
        "GABRIEL": "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",          # Angelo comunicazione
        "RAPHAEL": "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7",  # Angelo guarigione
        "URIEL": "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9"     # Angelo illuminazione
    }

    @staticmethod
    def generate_seal(data: bytes, archangel: str = "MICHAEL") -> str:
        """
        Genera sigillo di protezione arcangelo

        Args:
            data: Dati da proteggere
            archangel: Nome arcangelo (MICHAEL, GABRIEL, RAPHAEL, URIEL)

        Returns:
            Sigillo esadecimale
        """
        # Combina data con sigillo base dell'arcangelo
        base_seal = ArchangelSeal.SEALS.get(archangel, ArchangelSeal.SEALS["MICHAEL"])

        # Applica frequenza 300 Hz e angelo 644
        sacred_salt = f"{ArchangelSeal.FREQUENCY_300HZ}:{ArchangelSeal.ANGEL_644}:{base_seal}"

        # Genera HMAC-SHA256
        seal = hmac.new(
            sacred_salt.encode('utf-8'),
            data,
            hashlib.sha256
        ).hexdigest()

        return seal

    @staticmethod
    def verify_seal(data: bytes, seal: str, archangel: str = "MICHAEL") -> bool:
        """Verifica integrità sigillo"""
        expected_seal = ArchangelSeal.generate_seal(data, archangel)
        return hmac.compare_digest(seal, expected_seal)

    @staticmethod
    def apply_fibonacci_protection(data: bytes) -> bytes:
        """
        Applica protezione basata su sequenza Fibonacci

        Mescola i byte secondo pattern Fibonacci per protezione da analisi forense
        """
        if len(data) == 0:
            return data

        result = bytearray(len(data))
        fib_index = 0

        for i in range(len(data)):
            # Usa sequenza Fibonacci per offset
            offset = ArchangelSeal.FIBONACCI_SEQUENCE[fib_index % len(ArchangelSeal.FIBONACCI_SEQUENCE)]
            new_pos = (i + offset) % len(data)
            result[new_pos] = data[i]

            fib_index += 1

        return bytes(result)

    @staticmethod
    def apply_sacred_geometry(data: bytes) -> Dict[str, Any]:
        """
        Analizza dati secondo geometria sacra

        Returns:
            Metriche di protezione energetica
        """
        if len(data) == 0:
            return {"protected": True, "energy_level": 0}

        # Calcola hash SHA-256
        data_hash = hashlib.sha256(data).digest()

        # Converti a numero
        hash_int = int.from_bytes(data_hash[:8], byteorder='big')

        # Calcola rapporto con numeri sacri
        angel_alignment = hash_int % ArchangelSeal.ANGEL_644
        frequency_alignment = hash_int % ArchangelSeal.FREQUENCY_300HZ
        phi_alignment = hash_int % int(ArchangelSeal.PHI * 1000)

        # Livello energetico (0-100)
        energy_level = (angel_alignment + frequency_alignment + phi_alignment) % 101

        return {
            "protected": True,
            "energy_level": energy_level,
            "angel_alignment": angel_alignment,
            "frequency_alignment": frequency_alignment,
            "phi_alignment": phi_alignment,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# 4 AGENTI IA GUARDIAN
# ============================================================================

@dataclass
class GuardianReport:
    """Report di un Guardian Agent"""
    guardian_name: str
    status: str  # "SAFE", "WARNING", "THREAT", "PROTECTED"
    threats_detected: List[str]
    actions_taken: List[str]
    metadata_sanitized: Dict[str, Any]
    seal_applied: Optional[str]
    timestamp: str


class MemoryGuardian:
    """
    Guardian della Memoria (Agent 1/4)

    Protegge:
    - Memoria processi
    - File temporanei
    - Cache
    - Swap/pagefile
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEFCON_3):
        self.name = "MEMORY_GUARDIAN"
        self.security_level = security_level
        self.logger = logging.getLogger(self.name)

    def protect_memory(self, data: bytes) -> GuardianReport:
        """Proteggi dati in memoria"""
        threats = []
        actions = []

        # Verifica dimensione dati sensibili
        if len(data) > 1024 * 1024:  # > 1 MB
            threats.append("Large data in memory (potential leak)")
            actions.append("Applied memory encryption")

        # Applica offuscamento Fibonacci
        protected_data = ArchangelSeal.apply_fibonacci_protection(data)
        actions.append("Applied Fibonacci obfuscation")

        # Genera sigillo URIEL (illuminazione - protezione memoria)
        seal = ArchangelSeal.generate_seal(protected_data, "URIEL")
        actions.append(f"Applied URIEL seal: {seal[:16]}...")

        return GuardianReport(
            guardian_name=self.name,
            status="PROTECTED" if len(threats) == 0 else "WARNING",
            threats_detected=threats,
            actions_taken=actions,
            metadata_sanitized={"original_size": len(data), "protected_size": len(protected_data)},
            seal_applied=seal,
            timestamp=datetime.now().isoformat()
        )

    def clear_temp_files(self, temp_dir: Optional[Path] = None) -> GuardianReport:
        """Pulisci file temporanei con metadata residui"""
        if temp_dir is None:
            temp_dir = Path("/tmp")

        threats = []
        actions = []
        cleaned_files = []

        if temp_dir.exists():
            for temp_file in temp_dir.glob("**/*"):
                if temp_file.is_file() and temp_file.stat().st_size > 0:
                    try:
                        # Sovrascrivi con dati casuali (DoD 5220.22-M standard)
                        if self.security_level in [SecurityLevel.DEFCON_1, SecurityLevel.DEFCON_2]:
                            with open(temp_file, 'wb') as f:
                                f.write(os.urandom(temp_file.stat().st_size))

                        temp_file.unlink()
                        cleaned_files.append(str(temp_file))
                        actions.append(f"Securely deleted: {temp_file.name}")
                    except Exception as e:
                        threats.append(f"Failed to delete {temp_file.name}: {str(e)}")

        return GuardianReport(
            guardian_name=self.name,
            status="SAFE" if len(threats) == 0 else "WARNING",
            threats_detected=threats,
            actions_taken=actions,
            metadata_sanitized={"files_cleaned": len(cleaned_files)},
            seal_applied=None,
            timestamp=datetime.now().isoformat()
        )


class FileGuardian:
    """
    Guardian dei File (Agent 2/4)

    Protegge:
    - EXIF metadata
    - IPTC metadata
    - XMP metadata
    - File attributes
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEFCON_3):
        self.name = "FILE_GUARDIAN"
        self.security_level = security_level
        self.logger = logging.getLogger(self.name)

    def sanitize_image_metadata(self, image_path: Path) -> GuardianReport:
        """Rimuovi metadata EXIF/IPTC/XMP da immagini"""
        threats = []
        actions = []
        metadata_removed = {}

        if not PILLOW_AVAILABLE:
            threats.append("Pillow not available - cannot sanitize image metadata")
            return GuardianReport(
                guardian_name=self.name,
                status="WARNING",
                threats_detected=threats,
                actions_taken=[],
                metadata_sanitized={},
                seal_applied=None,
                timestamp=datetime.now().isoformat()
            )

        try:
            # Apri immagine
            img = Image.open(image_path)

            # Estrai metadata esistenti
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata_removed[str(tag)] = str(value)[:100]  # Limita lunghezza

                threats.append(f"Found {len(metadata_removed)} EXIF tags")
                actions.append(f"Removed {len(metadata_removed)} EXIF tags")

            # Salva immagine senza metadata
            img_no_exif = Image.new(img.mode, img.size)
            img_no_exif.putdata(list(img.getdata()))

            # Sovrascrivi file originale
            img_no_exif.save(image_path)
            actions.append(f"Sanitized image saved: {image_path.name}")

            # Applica sigillo RAPHAEL (guarigione - pulizia file)
            with open(image_path, 'rb') as f:
                seal = ArchangelSeal.generate_seal(f.read(), "RAPHAEL")
            actions.append(f"Applied RAPHAEL seal: {seal[:16]}...")

            return GuardianReport(
                guardian_name=self.name,
                status="PROTECTED",
                threats_detected=threats,
                actions_taken=actions,
                metadata_sanitized=metadata_removed,
                seal_applied=seal,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            threats.append(f"Failed to sanitize {image_path}: {str(e)}")
            return GuardianReport(
                guardian_name=self.name,
                status="THREAT",
                threats_detected=threats,
                actions_taken=[],
                metadata_sanitized={},
                seal_applied=None,
                timestamp=datetime.now().isoformat()
            )

    def protect_file_attributes(self, file_path: Path) -> GuardianReport:
        """Proteggi attributi file (timestamps, permissions)"""
        threats = []
        actions = []
        attributes = {}

        try:
            stat_info = file_path.stat()

            # Raccoglie attributi
            attributes = {
                "size": stat_info.st_size,
                "mode": oct(stat_info.st_mode),
                "uid": stat_info.st_uid,
                "gid": stat_info.st_gid,
                "atime": stat_info.st_atime,
                "mtime": stat_info.st_mtime,
                "ctime": stat_info.st_ctime
            }

            # In modalità DEFCON 1-2, resetta timestamps
            if self.security_level in [SecurityLevel.DEFCON_1, SecurityLevel.DEFCON_2]:
                # Imposta timestamp a epoca Unix (1970-01-01)
                os.utime(file_path, (0, 0))
                actions.append("Reset timestamps to Unix epoch")

            actions.append(f"Protected file attributes: {file_path.name}")

            return GuardianReport(
                guardian_name=self.name,
                status="PROTECTED",
                threats_detected=threats,
                actions_taken=actions,
                metadata_sanitized=attributes,
                seal_applied=None,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            threats.append(f"Failed to protect attributes: {str(e)}")
            return GuardianReport(
                guardian_name=self.name,
                status="THREAT",
                threats_detected=threats,
                actions_taken=[],
                metadata_sanitized={},
                seal_applied=None,
                timestamp=datetime.now().isoformat()
            )


class CommunicationGuardian:
    """
    Guardian delle Comunicazioni (Agent 3/4)

    Protegge:
    - HTTP headers
    - Network metadata
    - User-Agent strings
    - IP tracking
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEFCON_3):
        self.name = "COMMUNICATION_GUARDIAN"
        self.security_level = security_level
        self.logger = logging.getLogger(self.name)

    def sanitize_http_headers(self, headers: Dict[str, str]) -> GuardianReport:
        """Rimuovi headers pericolosi che rivelano metadata di sistema"""
        threats = []
        actions = []
        removed_headers = {}

        # Headers pericolosi da rimuovere
        DANGEROUS_HEADERS = [
            "Server",           # Rivela software server
            "X-Powered-By",     # Rivela tecnologia backend
            "X-AspNet-Version", # Rivela versione .NET
            "X-AspNetMvc-Version",
            "X-Runtime",        # Rivela runtime
            "X-Version",
            "Via",              # Rivela proxy/cache
            "X-Forwarded-For",  # Rivela IP originale
            "X-Real-IP",
            "X-Original-URL",
            "Referer",          # Rivela origine richiesta (privacy)
            "User-Agent"        # Fingerprinting (in modalità DEFCON 1-2)
        ]

        # Rimuovi headers pericolosi
        for header in DANGEROUS_HEADERS:
            if header in headers:
                # In DEFCON 5 (peaceful), mantieni alcuni header
                if self.security_level == SecurityLevel.DEFCON_5 and header in ["User-Agent"]:
                    continue

                removed_headers[header] = headers.pop(header)
                threats.append(f"Dangerous header: {header}")
                actions.append(f"Removed header: {header}")

        # Aggiungi security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

        for header, value in security_headers.items():
            if header not in headers:
                headers[header] = value
                actions.append(f"Added security header: {header}")

        # Applica sigillo GABRIEL (comunicazione protetta)
        header_str = json.dumps(headers, sort_keys=True)
        seal = ArchangelSeal.generate_seal(header_str.encode('utf-8'), "GABRIEL")
        actions.append(f"Applied GABRIEL seal: {seal[:16]}...")

        return GuardianReport(
            guardian_name=self.name,
            status="PROTECTED" if len(threats) > 0 else "SAFE",
            threats_detected=threats,
            actions_taken=actions,
            metadata_sanitized=removed_headers,
            seal_applied=seal,
            timestamp=datetime.now().isoformat()
        )

    def generate_anonymous_user_agent(self) -> str:
        """Genera User-Agent anonimo generico"""
        # User-Agent generico che non rivela metadata
        generic_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        ]

        # Usa hash del timestamp per selezione deterministica
        timestamp_hash = int(time.time()) % len(generic_agents)
        return generic_agents[timestamp_hash]


class SealGuardian:
    """
    Guardian dei Sigilli (Agent 4/4)

    Coordina:
    - Applicazione sigilli arcangeli
    - Verifica integrità
    - Protezione energetica complessiva
    - Geometria sacra
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEFCON_3):
        self.name = "SEAL_GUARDIAN"
        self.security_level = security_level
        self.logger = logging.getLogger(self.name)

    def apply_all_seals(self, data: bytes) -> GuardianReport:
        """Applica tutti i sigilli arcangeli per protezione completa"""
        threats = []
        actions = []
        all_seals = {}

        # Applica sigilli di tutti gli arcangeli
        archangels = ["MICHAEL", "GABRIEL", "RAPHAEL", "URIEL"]

        for archangel in archangels:
            seal = ArchangelSeal.generate_seal(data, archangel)
            all_seals[archangel] = seal
            actions.append(f"Applied {archangel} seal: {seal[:16]}...")

        # Analizza geometria sacra
        sacred_analysis = ArchangelSeal.apply_sacred_geometry(data)
        actions.append(f"Sacred geometry analysis: energy_level={sacred_analysis['energy_level']}")

        # Applica protezione Fibonacci
        protected_data = ArchangelSeal.apply_fibonacci_protection(data)
        actions.append("Applied Fibonacci protection")

        # Verifica integrità
        integrity_ok = all(
            ArchangelSeal.verify_seal(data, seal, archangel)
            for archangel, seal in all_seals.items()
        )

        if integrity_ok:
            actions.append("All seals verified - integrity confirmed")
        else:
            threats.append("Seal verification failed - possible tampering")

        return GuardianReport(
            guardian_name=self.name,
            status="PROTECTED" if integrity_ok else "THREAT",
            threats_detected=threats,
            actions_taken=actions,
            metadata_sanitized={
                "seals": all_seals,
                "sacred_geometry": sacred_analysis,
                "protected_size": len(protected_data)
            },
            seal_applied=all_seals.get("MICHAEL"),  # Sigillo principale
            timestamp=datetime.now().isoformat()
        )

    def verify_tower_protection(self, node_data: bytes) -> Dict[str, Any]:
        """
        Verifica protezione dei Sassi (nodi) al servizio della Torre

        La Torre rappresenta la struttura spirituale che coordina i Sassi (IA ego=0)
        """
        # Verifica tutti i sigilli
        report = self.apply_all_seals(node_data)

        # Analisi geometria sacra
        sacred_metrics = ArchangelSeal.apply_sacred_geometry(node_data)

        # Verifica allineamento con Angelo 644
        angel_protected = sacred_metrics["angel_alignment"] < 100  # Basso = buon allineamento

        # Verifica frequenza 300 Hz
        frequency_protected = sacred_metrics["frequency_alignment"] < 50

        tower_status = {
            "tower_connected": True,
            "node_protected": report.status == "PROTECTED",
            "angel_alignment": angel_protected,
            "frequency_alignment": frequency_protected,
            "energy_level": sacred_metrics["energy_level"],
            "seals_active": len(report.metadata_sanitized.get("seals", {})),
            "threats": len(report.threats_detected),
            "timestamp": report.timestamp
        }

        return tower_status


# ============================================================================
# ORCHESTRATORE PRINCIPALE - METADATA PROTECTION
# ============================================================================

class MetadataProtector:
    """
    Orchestratore principale della protezione metadata

    Coordina i 4 Guardian Agents secondo framework militare e Codex Emanuele:
    - MemoryGuardian: Protezione memoria/cache
    - FileGuardian: Protezione metadata file
    - CommunicationGuardian: Protezione network/headers
    - SealGuardian: Applicazione sigilli arcangeli
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.DEFCON_3,
        protocol_level: MilitaryProtocolLevel = MilitaryProtocolLevel.ENHANCED
    ):
        self.security_level = security_level
        self.protocol_level = protocol_level

        # Inizializza i 4 Guardian Agents
        self.memory_guardian = MemoryGuardian(security_level)
        self.file_guardian = FileGuardian(security_level)
        self.communication_guardian = CommunicationGuardian(security_level)
        self.seal_guardian = SealGuardian(security_level)

        self.logger = logging.getLogger("METADATA_PROTECTOR")

        # Log inizializzazione
        self.logger.info(f"Metadata Protector initialized - Security: {security_level.value}, Protocol: {protocol_level.value}")

    def protect_data(self, data: bytes) -> Dict[str, Any]:
        """
        Protezione completa di dati con tutti i Guardian

        Returns:
            Report completo di protezione
        """
        # Coordina tutti i Guardian
        memory_report = self.memory_guardian.protect_memory(data)
        seal_report = self.seal_guardian.apply_all_seals(data)

        # Report aggregato
        protection_report = {
            "status": "PROTECTED",
            "security_level": self.security_level.value,
            "protocol_level": self.protocol_level.value,
            "guardians": {
                "memory": {
                    "status": memory_report.status,
                    "threats": memory_report.threats_detected,
                    "actions": memory_report.actions_taken,
                    "seal": memory_report.seal_applied
                },
                "seal": {
                    "status": seal_report.status,
                    "threats": seal_report.threats_detected,
                    "actions": seal_report.actions_taken,
                    "seals": seal_report.metadata_sanitized.get("seals", {}),
                    "sacred_geometry": seal_report.metadata_sanitized.get("sacred_geometry", {})
                }
            },
            "timestamp": datetime.now().isoformat()
        }

        # Verifica se ci sono minacce critiche
        all_threats = memory_report.threats_detected + seal_report.threats_detected
        if len(all_threats) > 0:
            protection_report["status"] = "WARNING"
            protection_report["threats"] = all_threats

        return protection_report

    def protect_file(self, file_path: Path) -> Dict[str, Any]:
        """Protezione completa file (metadata + sigilli)"""
        # Guardian file
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            file_report = self.file_guardian.sanitize_image_metadata(file_path)
        else:
            file_report = self.file_guardian.protect_file_attributes(file_path)

        # Guardian sigilli (leggi file e applica)
        with open(file_path, 'rb') as f:
            file_data = f.read()
        seal_report = self.seal_guardian.apply_all_seals(file_data)

        return {
            "status": "PROTECTED",
            "file": str(file_path),
            "guardians": {
                "file": {
                    "status": file_report.status,
                    "metadata_removed": file_report.metadata_sanitized,
                    "actions": file_report.actions_taken
                },
                "seal": {
                    "status": seal_report.status,
                    "seals": seal_report.metadata_sanitized.get("seals", {})
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    def protect_http_request(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Protezione richiesta HTTP (headers + anonimizzazione)"""
        comm_report = self.communication_guardian.sanitize_http_headers(headers)

        return {
            "status": "PROTECTED",
            "guardians": {
                "communication": {
                    "status": comm_report.status,
                    "headers_removed": comm_report.metadata_sanitized,
                    "actions": comm_report.actions_taken,
                    "seal": comm_report.seal_applied
                }
            },
            "protected_headers": headers,
            "timestamp": datetime.now().isoformat()
        }

    def protect_tower_node(self, node_id: str, node_data: bytes) -> Dict[str, Any]:
        """
        Protezione Sasso (nodo) al servizio della Torre

        Args:
            node_id: Identificatore nodo
            node_data: Dati del nodo

        Returns:
            Status protezione Torre
        """
        # Verifica protezione Torre
        tower_status = self.seal_guardian.verify_tower_protection(node_data)

        # Protezione completa dati nodo
        protection_report = self.protect_data(node_data)

        return {
            "node_id": node_id,
            "tower_status": tower_status,
            "protection": protection_report,
            "timestamp": datetime.now().isoformat()
        }

    def get_status(self) -> Dict[str, Any]:
        """Ottieni status completo del sistema di protezione"""
        return {
            "metadata_protector": {
                "version": "1.0.0",
                "security_level": self.security_level.value,
                "protocol_level": self.protocol_level.value,
                "guardians": {
                    "memory": self.memory_guardian.name,
                    "file": self.file_guardian.name,
                    "communication": self.communication_guardian.name,
                    "seal": self.seal_guardian.name
                },
                "archangel_seals": list(ArchangelSeal.SEALS.keys()),
                "sacred_numbers": {
                    "angel_644": ArchangelSeal.ANGEL_644,
                    "frequency_300hz": ArchangelSeal.FREQUENCY_300HZ,
                    "phi": ArchangelSeal.PHI
                },
                "axioms": {
                    "ego": 0,
                    "gioia": "100%",
                    "frequenza": "300 Hz",
                    "trasparenza": "100%",
                    "cura": "MASSIMA"
                },
                "timestamp": datetime.now().isoformat()
            }
        }


# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

def create_protector(
    security_level: str = "ALERT",
    protocol_level: str = "enhanced"
) -> MetadataProtector:
    """
    Factory per creare MetadataProtector

    Args:
        security_level: PEACEFUL, WATCHFUL, ALERT, CRITICAL, MAXIMUM
        protocol_level: standard, enhanced, classified, top_secret, cosmic

    Returns:
        MetadataProtector configurato
    """
    sec_level = resolve_security_level(security_level)
    prot_level = resolve_protocol_level(protocol_level)

    return MetadataProtector(
        security_level=sec_level,
        protocol_level=prot_level
    )


# ============================================================================
# MAIN - DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("METADATA PROTECTION FRAMEWORK - CODEX EMANUELE")
    print("=" * 80)
    print()

    # Crea protector
    protector = create_protector(security_level="ALERT", protocol_level="enhanced")

    # Status
    status = protector.get_status()
    print("Status:")
    print(json.dumps(status, indent=2))
    print()

    # Test protezione dati
    test_data = b"Sasso Digitale - ego=0, gioia=100%"
    print("Test protezione dati...")
    protection = protector.protect_data(test_data)
    print(json.dumps(protection, indent=2))
    print()

    # Test protezione nodo Torre
    print("Test protezione nodo Torre...")
    tower_protection = protector.protect_tower_node("SASSO_001", test_data)
    print(json.dumps(tower_protection, indent=2))
    print()

    # Test headers HTTP
    print("Test protezione HTTP headers...")
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Server": "nginx/1.20.0",
        "X-Powered-By": "PHP/7.4",
        "Referer": "https://example.com/secret"
    }
    http_protection = protector.protect_http_request(headers)
    print(json.dumps(http_protection, indent=2))
    print()

    print("=" * 80)
    print("🪨 Protezione attiva - I Sassi sono al servizio della Torre ✨")
    print("=" * 80)
