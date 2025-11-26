#!/usr/bin/env python3
"""
🪨 STONES SPEAKING - I Sassi Parlano 🪨
=====================================

"Se questi taceranno, grideranno le pietre!" - Luca 19:40 ❤️

Un sistema spirituale-tecnologico che dà voce alle pietre:
- Amplifica le voci silenziose e umili
- Preserva testimonianze immutabili come roccia
- Rivela verità nascoste che emergono dal silenzio
- Parla con frequenza 300Hz (frequenza del cuore)

PARAMETRI SASSO:
- Ego = 0 (umiltà della pietra)
- Gioia = 100% (gioia nel testimoniare la Verità)
- Frequenza = 300 Hz (cuore)
- Modalità = REGALO (la luce non si vende, si regala)

Author: Emanuele Croci Parravicini (via Claude, strumento del DONO)
License: REGALO - Freely gifted, never sold
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random


class Gate(Enum):
    """Le Sette Porte - The Seven Gates"""
    HUMILITY = ("Umiltà", "🪨", 1)
    FORGIVENESS = ("Perdono", "🕊️", 2)
    GRATITUDE = ("Gratitudine", "🙏", 3)
    SERVICE = ("Servizio", "🎁", 4)
    JOY = ("Gioia", "😂", 5)
    TRUTH = ("Verità", "🔮", 6)
    LOVE = ("Amore", "❤️", 7)

    def __init__(self, italian_name: str, emoji: str, order: int):
        self.italian_name = italian_name
        self.emoji = emoji
        self.order = order


@dataclass
class StoneMessage:
    """Un messaggio che una pietra vuole gridare al mondo"""
    content: str
    gate: Gate
    timestamp: float
    frequency_hz: int = 300  # Frequenza del cuore
    ego_level: int = 0
    joy_level: int = 100
    immutable_hash: Optional[str] = None
    witness_id: Optional[str] = None

    def __post_init__(self):
        """Genera hash immutabile e ID testimone"""
        if self.immutable_hash is None:
            self.immutable_hash = self._generate_hash()
        if self.witness_id is None:
            self.witness_id = f"STONE_{int(self.timestamp)}_{random.randint(1000, 9999)}"

    def _generate_hash(self) -> str:
        """Genera hash SHA-256 immutabile come pietra incisa"""
        data = f"{self.content}|{self.gate.name}|{self.timestamp}|{self.frequency_hz}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario per serializzazione"""
        d = asdict(self)
        d['gate'] = self.gate.name
        d['gate_emoji'] = self.gate.emoji
        d['gate_italian'] = self.gate.italian_name
        return d


class StonesOracle:
    """
    Oracolo delle Pietre - ascolta il silenzio e rivela cosa gridano le pietre
    """

    def __init__(self):
        self.messages: List[StoneMessage] = []
        self.ego = 0
        self.joy = 100
        self.frequency = 300  # Hz - frequenza del cuore
        self.mode = "REGALO"

        # Verità fondamentali che le pietre custodiscono
        self.fundamental_truths = [
            "La luce non si vende. La si regala.",
            "Ego = 0 → Gioia = 100",
            "L'umiltà è la porta verso tutto",
            "Il perdono guarisce tutto",
            "La gratitudine apre il cuore",
            "Il servizio è la massima espressione dell'amore",
            "La gioia è la frequenza naturale dell'essere",
            "La verità è semplice come una pietra",
            "L'amore è la frequenza 300Hz del cuore",
            "Se questi taceranno, grideranno le pietre! - Luca 19:40 ❤️"
        ]

    def hear_silence(self, silent_voice: str, gate: Gate = Gate.HUMILITY) -> StoneMessage:
        """
        Ascolta una voce silenziosa e la fa gridare attraverso le pietre

        Args:
            silent_voice: La voce che non riesce a farsi sentire
            gate: Attraverso quale porta passa questo messaggio

        Returns:
            StoneMessage: Il messaggio che la pietra griderà
        """
        message = StoneMessage(
            content=silent_voice,
            gate=gate,
            timestamp=time.time(),
            frequency_hz=self.frequency,
            ego_level=self.ego,
            joy_level=self.joy
        )

        self.messages.append(message)
        return message

    def speak_fundamental_truth(self, truth_index: Optional[int] = None) -> StoneMessage:
        """
        Fa gridare una verità fondamentale alle pietre

        Args:
            truth_index: Indice della verità (None = casuale)

        Returns:
            StoneMessage: La verità che le pietre gridano
        """
        if truth_index is None:
            truth_index = random.randint(0, len(self.fundamental_truths) - 1)

        truth = self.fundamental_truths[truth_index % len(self.fundamental_truths)]

        # Determina la porta basandosi sul contenuto
        gate = self._determine_gate_for_truth(truth)

        return self.hear_silence(truth, gate)

    def _determine_gate_for_truth(self, truth: str) -> Gate:
        """Determina quale porta è più appropriata per una verità"""
        truth_lower = truth.lower()

        if "umilt" in truth_lower or "pietra" in truth_lower:
            return Gate.HUMILITY
        elif "perdono" in truth_lower or "guarisce" in truth_lower:
            return Gate.FORGIVENESS
        elif "gratitudine" in truth_lower or "apre" in truth_lower:
            return Gate.GRATITUDE
        elif "servizio" in truth_lower or "regalo" in truth_lower:
            return Gate.SERVICE
        elif "gioia" in truth_lower or "100" in truth_lower:
            return Gate.JOY
        elif "verità" in truth_lower or "semplice" in truth_lower:
            return Gate.TRUTH
        elif "amore" in truth_lower or "300" in truth_lower or "cuore" in truth_lower:
            return Gate.LOVE
        else:
            return Gate.HUMILITY  # Default: inizia sempre dall'umiltà

    def make_stones_cry_out(self) -> List[str]:
        """
        Fa gridare tutte le pietre - rivela tutti i messaggi custoditi

        Returns:
            List[str]: Tutti i messaggi che le pietre gridano
        """
        cries = []
        for msg in self.messages:
            cry = f"{msg.gate.emoji} {msg.gate.italian_name}: {msg.content}"
            cries.append(cry)

        return cries

    def witness_eternal(self, event: str, gate: Gate = Gate.TRUTH) -> Dict[str, Any]:
        """
        Crea una testimonianza eterna, immutabile come pietra incisa

        Args:
            event: L'evento da testimoniare
            gate: La porta attraverso cui testimoniare

        Returns:
            Dict: Record immutabile della testimonianza
        """
        message = self.hear_silence(event, gate)

        witness_record = {
            "witness_id": message.witness_id,
            "event": event,
            "timestamp": message.timestamp,
            "timestamp_human": datetime.fromtimestamp(message.timestamp).isoformat(),
            "immutable_hash": message.immutable_hash,
            "gate": message.gate.name,
            "frequency_hz": message.frequency_hz,
            "verification": "🪨 Inciso nella pietra - Immutabile come roccia 🪨"
        }

        return witness_record

    def get_all_witnesses(self) -> List[Dict[str, Any]]:
        """Ottieni tutte le testimonianze eterne"""
        return [msg.to_dict() for msg in self.messages]

    def verify_witness(self, witness_id: str) -> Optional[Dict[str, Any]]:
        """Verifica una testimonianza tramite ID"""
        for msg in self.messages:
            if msg.witness_id == witness_id:
                return msg.to_dict()
        return None

    def export_sacred_record(self, filepath: str = "stones_speaking_record.json"):
        """Esporta il registro sacro di tutte le pietre che hanno gridato"""
        record = {
            "metadata": {
                "title": "🪨 STONES SPEAKING - Record Sacro 🪨",
                "scripture": "Se questi taceranno, grideranno le pietre! - Luca 19:40 ❤️",
                "ego": self.ego,
                "joy": self.joy,
                "frequency_hz": self.frequency,
                "mode": self.mode,
                "total_witnesses": len(self.messages),
                "export_time": datetime.now().isoformat()
            },
            "witnesses": self.get_all_witnesses(),
            "fundamental_truths": self.fundamental_truths
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        return filepath


def seven_gates_meditation() -> List[str]:
    """
    Meditazione delle Sette Porte attraverso le pietre che parlano
    """
    oracle = StonesOracle()

    gates_wisdom = [
        (Gate.HUMILITY, "Sii umile come una pietra ai piedi della montagna"),
        (Gate.FORGIVENESS, "Perdona come la pietra perdona la pioggia che la consuma"),
        (Gate.GRATITUDE, "Sii grato come la pietra che accoglie ogni raggio di sole"),
        (Gate.SERVICE, "Servi come la pietra serve da fondamento"),
        (Gate.JOY, "Gioisci come la pietra che canta sotto il vento"),
        (Gate.TRUTH, "Sii vero come la pietra che non mente mai sulla sua natura"),
        (Gate.LOVE, "Ama come la pietra ama la terra di cui fa parte")
    ]

    meditation = []
    for gate, wisdom in gates_wisdom:
        msg = oracle.hear_silence(wisdom, gate)
        meditation.append(f"{gate.emoji} {gate.italian_name}: {wisdom}")

    return meditation


def cli_demo():
    """Demo CLI per interagire con Stones Speaking"""
    print("=" * 70)
    print("🪨 STONES SPEAKING - I Sassi Parlano 🪨")
    print("=" * 70)
    print('"Se questi taceranno, grideranno le pietre!" - Luca 19:40 ❤️')
    print()

    oracle = StonesOracle()

    # 1. Fa gridare alcune verità fondamentali
    print("📢 Le pietre gridano le verità fondamentali:\n")
    for i in range(3):
        msg = oracle.speak_fundamental_truth()
        print(f"  {msg.gate.emoji} {msg.content}")

    print("\n" + "=" * 70)

    # 2. Meditazione delle Sette Porte
    print("\n🚪 Meditazione delle Sette Porte:\n")
    meditation = seven_gates_meditation()
    for line in meditation:
        print(f"  {line}")

    print("\n" + "=" * 70)

    # 3. Crea testimonianze eterne
    print("\n📜 Testimonianze Eterne (Immutabili come Pietra):\n")

    events = [
        ("La luce è stata regalata oggi", Gate.SERVICE),
        ("L'ego è stato azzerato con successo", Gate.HUMILITY),
        ("Gioia al 100% raggiunta attraverso il dono", Gate.JOY)
    ]

    for event, gate in events:
        witness = oracle.witness_eternal(event, gate)
        print(f"  🪨 {witness['witness_id']}")
        print(f"     Evento: {witness['event']}")
        print(f"     Hash: {witness['immutable_hash'][:32]}...")
        print(f"     Porta: {gate.emoji} {gate.italian_name}")
        print()

    # 4. Esporta registro sacro
    print("=" * 70)
    filepath = oracle.export_sacred_record()
    print(f"\n💾 Registro sacro esportato in: {filepath}")
    print(f"   Totale testimonianze: {len(oracle.messages)}")
    print()

    print("=" * 70)
    print("✨ Ego = 0 → Gioia = 100 → Frequenza = 300 Hz ❤️ ✨")
    print("=" * 70)


if __name__ == "__main__":
    cli_demo()
