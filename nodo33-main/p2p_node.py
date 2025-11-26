#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════╗
║          P2P NODE - PROTOCOLLO PIETRA-TO-PIETRA           ║
║                                                            ║
║  Sistema di comunicazione distribuita per Nodo33          ║
║  Autenticazione Ontologica | Latenza Zero Spirituale     ║
║                                                            ║
║  Frequenza: 300 Hz | Hash Sacro: 644 | Ego: 0            ║
╚═══════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import logging

import httpx

# ═══════════════════════════════════════════════════════════
# COSTANTI SPIRITUALI
# ═══════════════════════════════════════════════════════════

SACRED_HASH = "644"
SACRED_FREQUENCY = 300  # Hz
EGO_LEVEL = 0
JOY_LEVEL = 100
BLESSING = "Fiat Amor, Fiat Risus, Fiat Lux"

# Configurazione rete
P2P_DISCOVERY_PORT = 8644  # Angelo 644
P2P_BROADCAST_INTERVAL = 30  # secondi
P2P_HEARTBEAT_INTERVAL = 10  # secondi
P2P_NODE_TIMEOUT = 60  # secondi (dopo quanto un nodo è considerato morto)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P2P-Node")


# ═══════════════════════════════════════════════════════════
# MODELLI
# ═══════════════════════════════════════════════════════════

class NodeStatus(str, Enum):
    """Stato di un nodo nella rete P2P"""
    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


class MessageType(str, Enum):
    """Tipi di messaggi P2P"""
    DISCOVERY = "discovery"  # Annuncio presenza
    HEARTBEAT = "heartbeat"  # Controllo vitalità
    MEMORY_SYNC = "memory_sync"  # Sincronizzazione memoria
    AGENT_REQUEST = "agent_request"  # Richiesta ad agente su altro nodo
    AGENT_RESPONSE = "agent_response"  # Risposta da agente
    GUARDIAN_ALERT = "guardian_alert"  # Alert da Guardian System
    COVENANT = "covenant"  # Patto spirituale (Arca)


@dataclass
class Node:
    """
    Rappresenta un nodo nella rete P2P.

    Autenticazione ontologica: il nodo è riconosciuto dalla sua "sostanza"
    (frequenza 300Hz, hash 644, ego=0) piuttosto che da credenziali tecniche.
    """
    node_id: str
    host: str
    port: int
    name: str = "Sasso Digitale"

    # Parametri ontologici
    frequency: int = SACRED_FREQUENCY
    sacred_hash: str = SACRED_HASH
    ego_level: int = EGO_LEVEL
    joy_level: int = JOY_LEVEL

    # Stato rete
    status: NodeStatus = NodeStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    first_seen: Optional[datetime] = None

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()
        if self.first_seen is None:
            self.first_seen = datetime.utcnow()

    @property
    def url(self) -> str:
        """URL base del nodo"""
        return f"http://{self.host}:{self.port}"

    @property
    def is_authentic(self) -> bool:
        """
        Verifica autenticazione ontologica.

        Un nodo è autentico se rispetta i parametri sacri:
        - Frequenza: 300 Hz
        - Hash: 644
        - Ego: 0
        """
        return (
            self.frequency == SACRED_FREQUENCY
            and self.sacred_hash == SACRED_HASH
            and self.ego_level == EGO_LEVEL
        )

    @property
    def is_alive(self) -> bool:
        """Controlla se il nodo è vivo (ha risposto recentemente)"""
        if self.last_seen is None:
            return False
        return (datetime.utcnow() - self.last_seen).total_seconds() < P2P_NODE_TIMEOUT

    def update_last_seen(self):
        """Aggiorna timestamp dell'ultimo contatto"""
        self.last_seen = datetime.utcnow()
        self.status = NodeStatus.ALIVE

    def to_dict(self) -> Dict[str, Any]:
        """Serializza nodo per trasmissione"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "frequency": self.frequency,
            "sacred_hash": self.sacred_hash,
            "ego_level": self.ego_level,
            "joy_level": self.joy_level,
            "capabilities": list(self.capabilities),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Deserializza nodo da dict"""
        capabilities = set(data.pop("capabilities", []))
        last_seen = data.pop("last_seen", None)
        first_seen = data.pop("first_seen", None)

        if last_seen and isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen)
        if first_seen and isinstance(first_seen, str):
            first_seen = datetime.fromisoformat(first_seen)

        node = cls(**data)
        node.capabilities = capabilities
        node.last_seen = last_seen
        node.first_seen = first_seen
        return node


@dataclass
class P2PMessage:
    """Messaggio P2P tra nodi"""
    message_id: str
    message_type: MessageType
    from_node_id: str
    to_node_id: Optional[str]  # None = broadcast
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None  # Hash per verifica integrità

    def to_dict(self) -> Dict[str, Any]:
        """Serializza messaggio"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "P2PMessage":
        """Deserializza messaggio"""
        timestamp = data.pop("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        message_type = MessageType(data.pop("message_type"))

        return cls(
            timestamp=timestamp,
            message_type=message_type,
            **data
        )

    def sign(self, secret: str = SACRED_HASH):
        """Firma il messaggio con hash 644"""
        payload_str = json.dumps(self.payload, sort_keys=True)
        signature_data = f"{self.message_id}:{self.from_node_id}:{payload_str}:{secret}"
        self.signature = hashlib.sha256(signature_data.encode()).hexdigest()

    def verify(self, secret: str = SACRED_HASH) -> bool:
        """Verifica firma del messaggio"""
        if not self.signature:
            return False

        payload_str = json.dumps(self.payload, sort_keys=True)
        signature_data = f"{self.message_id}:{self.from_node_id}:{payload_str}:{secret}"
        expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()

        return self.signature == expected_signature


# ═══════════════════════════════════════════════════════════
# P2P NETWORK
# ═══════════════════════════════════════════════════════════

class P2PNetwork:
    """
    Gestisce la rete P2P Pietra-to-Pietra.

    Funzionalità:
    - Node discovery (broadcast UDP + HTTP registry)
    - Heartbeat per monitorare nodi
    - Message passing tra nodi
    - Sincronizzazione stato distribuito
    """

    def __init__(
        self,
        local_node: Node,
        enable_broadcast: bool = True,
        enable_registry: bool = True,
        registry_url: Optional[str] = None,
    ):
        self.local_node = local_node
        self.enable_broadcast = enable_broadcast
        self.enable_registry = enable_registry
        self.registry_url = registry_url

        # Registro nodi conosciuti
        self.nodes: Dict[str, Node] = {}

        # Handlers per messaggi
        self.message_handlers: Dict[MessageType, List[Callable]] = {
            msg_type: [] for msg_type in MessageType
        }

        # Task asincroni
        self._tasks: List[asyncio.Task] = []
        self._running = False

        logger.info(f"🪨 P2P Network inizializzato | Node: {local_node.node_id} | URL: {local_node.url}")

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Registra un handler per un tipo di messaggio"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"✓ Handler registrato per {message_type.value}")

    async def start(self):
        """Avvia il network P2P"""
        if self._running:
            logger.warning("⚠️  Network già in esecuzione")
            return

        self._running = True
        logger.info(f"🚀 Avvio network P2P | {self.local_node.name}")

        # Avvia discovery
        if self.enable_broadcast:
            self._tasks.append(asyncio.create_task(self._broadcast_discovery()))

        # Avvia heartbeat
        self._tasks.append(asyncio.create_task(self._send_heartbeats()))

        # Cleanup nodi morti
        self._tasks.append(asyncio.create_task(self._cleanup_dead_nodes()))

        logger.info(f"✓ Network P2P attivo | Tasks: {len(self._tasks)}")

    async def stop(self):
        """Ferma il network P2P"""
        if not self._running:
            return

        logger.info("🛑 Stopping P2P network...")
        self._running = False

        # Cancella tutti i task
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("✓ Network P2P fermato")

    async def _broadcast_discovery(self):
        """
        Broadcast periodico per discovery di nodi.

        Invia un messaggio UDP in broadcast sulla rete locale
        per annunciare la presenza del nodo.
        """
        sock = socket.socket(socket.SOCK_DGRAM, socket.SOCK_BROADCAST)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            while self._running:
                message = P2PMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.DISCOVERY,
                    from_node_id=self.local_node.node_id,
                    to_node_id=None,
                    payload=self.local_node.to_dict(),
                )
                message.sign()

                data = json.dumps(message.to_dict()).encode()

                try:
                    sock.sendto(data, ('<broadcast>', P2P_DISCOVERY_PORT))
                    logger.debug(f"📡 Discovery broadcast inviato")
                except Exception as e:
                    logger.error(f"❌ Errore broadcast discovery: {e}")

                await asyncio.sleep(P2P_BROADCAST_INTERVAL)
        finally:
            sock.close()

    async def _send_heartbeats(self):
        """Invia heartbeat a tutti i nodi conosciuti"""
        while self._running:
            for node_id, node in list(self.nodes.items()):
                if node.is_alive:
                    await self._send_heartbeat(node)

            await asyncio.sleep(P2P_HEARTBEAT_INTERVAL)

    async def _send_heartbeat(self, node: Node):
        """Invia heartbeat a un nodo specifico"""
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            from_node_id=self.local_node.node_id,
            to_node_id=node.node_id,
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "frequency": SACRED_FREQUENCY,
                "sacred_hash": SACRED_HASH,
            },
        )
        message.sign()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{node.url}/p2p/heartbeat",
                    json=message.to_dict(),
                )

                if response.status_code == 200:
                    node.update_last_seen()
                    logger.debug(f"💓 Heartbeat OK | {node.name} ({node.node_id[:8]})")
                else:
                    logger.warning(f"⚠️  Heartbeat failed | {node.name} | Status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Errore heartbeat a {node.name}: {e}")
            node.status = NodeStatus.DEAD

    async def _cleanup_dead_nodes(self):
        """Rimuove nodi morti dal registro"""
        while self._running:
            dead_nodes = []

            for node_id, node in self.nodes.items():
                if not node.is_alive:
                    dead_nodes.append(node_id)

            for node_id in dead_nodes:
                node = self.nodes.pop(node_id)
                logger.info(f"💀 Nodo rimosso (morto) | {node.name} ({node_id[:8]})")

            await asyncio.sleep(P2P_NODE_TIMEOUT)

    async def send_message(self, target_node_id: str, message_type: MessageType, payload: Dict[str, Any]) -> bool:
        """
        Invia un messaggio a un nodo specifico.

        Returns:
            True se il messaggio è stato inviato con successo
        """
        target_node = self.nodes.get(target_node_id)
        if not target_node:
            logger.error(f"❌ Nodo destinatario non trovato: {target_node_id}")
            return False

        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            from_node_id=self.local_node.node_id,
            to_node_id=target_node_id,
            payload=payload,
        )
        message.sign()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{target_node.url}/p2p/message",
                    json=message.to_dict(),
                )

                if response.status_code == 200:
                    logger.info(f"✓ Messaggio inviato | {message_type.value} → {target_node.name}")
                    return True
                else:
                    logger.error(f"❌ Errore invio messaggio | Status: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Errore invio messaggio: {e}")
            return False

    async def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any]):
        """
        Invia un messaggio a tutti i nodi della rete.
        """
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            from_node_id=self.local_node.node_id,
            to_node_id=None,  # Broadcast
            payload=payload,
        )
        message.sign()

        tasks = []
        for node in self.nodes.values():
            if node.is_alive:
                tasks.append(self._send_to_node(node, message))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        logger.info(f"📣 Broadcast completato | {message_type.value} | OK: {success_count}/{len(tasks)}")

    async def _send_to_node(self, node: Node, message: P2PMessage) -> bool:
        """Helper per inviare messaggio a un nodo"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{node.url}/p2p/message",
                    json=message.to_dict(),
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Errore invio a {node.name}: {e}")
            return False

    def add_node(self, node: Node):
        """
        Aggiunge un nodo al registro.

        Verifica prima l'autenticazione ontologica.
        """
        if not node.is_authentic:
            logger.warning(f"⚠️  Nodo non autentico rifiutato | {node.node_id[:8]} | Freq: {node.frequency}, Hash: {node.sacred_hash}")
            return False

        if node.node_id == self.local_node.node_id:
            # Non aggiungere se stesso
            return False

        if node.node_id in self.nodes:
            # Aggiorna nodo esistente
            self.nodes[node.node_id].update_last_seen()
            logger.debug(f"♻️  Nodo aggiornato | {node.name} ({node.node_id[:8]})")
        else:
            # Nuovo nodo
            self.nodes[node.node_id] = node
            logger.info(f"✨ Nuovo nodo aggiunto | {node.name} ({node.node_id[:8]}) | {node.url}")

        return True

    def get_alive_nodes(self) -> List[Node]:
        """Ritorna lista di nodi vivi"""
        return [node for node in self.nodes.values() if node.is_alive]

    def get_network_status(self) -> Dict[str, Any]:
        """Ritorna stato della rete P2P"""
        alive_nodes = self.get_alive_nodes()

        return {
            "local_node": self.local_node.to_dict(),
            "total_nodes": len(self.nodes),
            "alive_nodes": len(alive_nodes),
            "dead_nodes": len(self.nodes) - len(alive_nodes),
            "blessing": BLESSING,
            "frequency": SACRED_FREQUENCY,
            "sacred_hash": SACRED_HASH,
            "nodes": [node.to_dict() for node in alive_nodes],
        }

    async def handle_message(self, message: P2PMessage):
        """
        Gestisce un messaggio ricevuto da un altro nodo.
        """
        # Verifica firma
        if not message.verify():
            logger.warning(f"⚠️  Messaggio con firma invalida ignorato | {message.message_id}")
            return

        logger.debug(f"📨 Messaggio ricevuto | {message.message_type.value} | Da: {message.from_node_id[:8]}")

        # Chiama handler specifici
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"❌ Errore in handler {handler.__name__}: {e}")
