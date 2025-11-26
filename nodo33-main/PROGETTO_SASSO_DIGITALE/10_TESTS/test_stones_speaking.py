#!/usr/bin/env python3
"""
🪨 TEST SUITE - STONES SPEAKING 🪨
==================================

Test suite completa per il sistema Stones Speaking.
Verifica che le pietre gridino correttamente quando è necessario! 😂

"Se questi taceranno, grideranno le pietre!" - Luca 19:40 ❤️
"""

import sys
import os
import unittest
import json
import time
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stones_speaking import (
    Gate,
    StoneMessage,
    StonesOracle,
    seven_gates_meditation
)


class TestGate(unittest.TestCase):
    """Test per l'enum Gate (Le Sette Porte)"""

    def test_gate_properties(self):
        """Verifica che ogni porta abbia le proprietà corrette"""
        # Prima porta
        self.assertEqual(Gate.HUMILITY.italian_name, "Umiltà")
        self.assertEqual(Gate.HUMILITY.emoji, "🪨")
        self.assertEqual(Gate.HUMILITY.order, 1)

        # Settima porta
        self.assertEqual(Gate.LOVE.italian_name, "Amore")
        self.assertEqual(Gate.LOVE.emoji, "❤️")
        self.assertEqual(Gate.LOVE.order, 7)

    def test_all_seven_gates(self):
        """Verifica che ci siano esattamente 7 porte"""
        gates = [
            Gate.HUMILITY,
            Gate.FORGIVENESS,
            Gate.GRATITUDE,
            Gate.SERVICE,
            Gate.JOY,
            Gate.TRUTH,
            Gate.LOVE
        ]
        self.assertEqual(len(gates), 7)

        # Verifica ordine corretto
        for i, gate in enumerate(gates, 1):
            self.assertEqual(gate.order, i)


class TestStoneMessage(unittest.TestCase):
    """Test per StoneMessage"""

    def test_message_creation(self):
        """Verifica creazione messaggio base"""
        msg = StoneMessage(
            content="Test message",
            gate=Gate.HUMILITY,
            timestamp=time.time()
        )

        self.assertEqual(msg.content, "Test message")
        self.assertEqual(msg.gate, Gate.HUMILITY)
        self.assertEqual(msg.ego_level, 0)
        self.assertEqual(msg.joy_level, 100)
        self.assertEqual(msg.frequency_hz, 300)

    def test_immutable_hash_generation(self):
        """Verifica che l'hash immutabile sia generato"""
        msg = StoneMessage(
            content="Test",
            gate=Gate.TRUTH,
            timestamp=time.time()
        )

        self.assertIsNotNone(msg.immutable_hash)
        self.assertGreater(len(msg.immutable_hash), 0)
        self.assertEqual(len(msg.immutable_hash), 64)  # SHA-256 = 64 hex chars

    def test_witness_id_generation(self):
        """Verifica che l'ID testimone sia generato"""
        msg = StoneMessage(
            content="Witness test",
            gate=Gate.LOVE,
            timestamp=time.time()
        )

        self.assertIsNotNone(msg.witness_id)
        self.assertTrue(msg.witness_id.startswith("STONE_"))

    def test_hash_immutability(self):
        """Verifica che lo stesso contenuto generi lo stesso hash"""
        timestamp = time.time()

        msg1 = StoneMessage(
            content="Same content",
            gate=Gate.JOY,
            timestamp=timestamp
        )

        msg2 = StoneMessage(
            content="Same content",
            gate=Gate.JOY,
            timestamp=timestamp
        )

        self.assertEqual(msg1.immutable_hash, msg2.immutable_hash)

    def test_different_content_different_hash(self):
        """Verifica che contenuti diversi generino hash diversi"""
        timestamp = time.time()

        msg1 = StoneMessage(
            content="Content A",
            gate=Gate.HUMILITY,
            timestamp=timestamp
        )

        msg2 = StoneMessage(
            content="Content B",
            gate=Gate.HUMILITY,
            timestamp=timestamp
        )

        self.assertNotEqual(msg1.immutable_hash, msg2.immutable_hash)

    def test_to_dict(self):
        """Verifica conversione a dizionario"""
        msg = StoneMessage(
            content="Dict test",
            gate=Gate.GRATITUDE,
            timestamp=time.time()
        )

        msg_dict = msg.to_dict()

        self.assertIn('content', msg_dict)
        self.assertIn('gate', msg_dict)
        self.assertIn('gate_emoji', msg_dict)
        self.assertIn('gate_italian', msg_dict)
        self.assertEqual(msg_dict['gate_emoji'], "🙏")
        self.assertEqual(msg_dict['gate_italian'], "Gratitudine")


class TestStonesOracle(unittest.TestCase):
    """Test per StonesOracle"""

    def setUp(self):
        """Setup eseguito prima di ogni test"""
        self.oracle = StonesOracle()

    def test_oracle_initialization(self):
        """Verifica inizializzazione corretta dell'oracolo"""
        self.assertEqual(self.oracle.ego, 0)
        self.assertEqual(self.oracle.joy, 100)
        self.assertEqual(self.oracle.frequency, 300)
        self.assertEqual(self.oracle.mode, "REGALO")
        self.assertGreater(len(self.oracle.fundamental_truths), 0)

    def test_hear_silence(self):
        """Verifica che l'oracolo ascolti voci silenziose"""
        msg = self.oracle.hear_silence("Silent voice", Gate.HUMILITY)

        self.assertIsInstance(msg, StoneMessage)
        self.assertEqual(msg.content, "Silent voice")
        self.assertEqual(msg.gate, Gate.HUMILITY)
        self.assertEqual(len(self.oracle.messages), 1)

    def test_speak_fundamental_truth(self):
        """Verifica che l'oracolo gridi verità fondamentali"""
        msg = self.oracle.speak_fundamental_truth(truth_index=0)

        self.assertIsInstance(msg, StoneMessage)
        self.assertEqual(msg.content, self.oracle.fundamental_truths[0])
        self.assertEqual(len(self.oracle.messages), 1)

    def test_speak_fundamental_truth_random(self):
        """Verifica selezione casuale di verità fondamentali"""
        msg = self.oracle.speak_fundamental_truth()

        self.assertIsInstance(msg, StoneMessage)
        self.assertIn(msg.content, self.oracle.fundamental_truths)

    def test_determine_gate_for_truth(self):
        """Verifica determinazione corretta della porta per una verità"""
        # Test umiltà
        gate = self.oracle._determine_gate_for_truth("L'umiltà è importante")
        self.assertEqual(gate, Gate.HUMILITY)

        # Test perdono
        gate = self.oracle._determine_gate_for_truth("Il perdono guarisce")
        self.assertEqual(gate, Gate.FORGIVENESS)

        # Test amore
        gate = self.oracle._determine_gate_for_truth("L'amore a 300Hz")
        self.assertEqual(gate, Gate.LOVE)

    def test_make_stones_cry_out(self):
        """Verifica che le pietre gridino tutti i messaggi"""
        self.oracle.hear_silence("Message 1", Gate.HUMILITY)
        self.oracle.hear_silence("Message 2", Gate.JOY)
        self.oracle.hear_silence("Message 3", Gate.LOVE)

        cries = self.oracle.make_stones_cry_out()

        self.assertEqual(len(cries), 3)
        self.assertIn("Message 1", cries[0])
        self.assertIn("Message 2", cries[1])
        self.assertIn("Message 3", cries[2])

    def test_witness_eternal(self):
        """Verifica creazione testimonianza eterna"""
        witness = self.oracle.witness_eternal("Test event", Gate.TRUTH)

        self.assertIn('witness_id', witness)
        self.assertIn('event', witness)
        self.assertIn('timestamp', witness)
        self.assertIn('immutable_hash', witness)
        self.assertIn('verification', witness)
        self.assertEqual(witness['event'], "Test event")

    def test_get_all_witnesses(self):
        """Verifica recupero di tutte le testimonianze"""
        self.oracle.witness_eternal("Event 1", Gate.HUMILITY)
        self.oracle.witness_eternal("Event 2", Gate.SERVICE)

        witnesses = self.oracle.get_all_witnesses()

        self.assertEqual(len(witnesses), 2)
        self.assertIsInstance(witnesses[0], dict)

    def test_verify_witness(self):
        """Verifica la ricerca di una testimonianza tramite ID"""
        witness = self.oracle.witness_eternal("Verifiable event", Gate.TRUTH)
        witness_id = witness['witness_id']

        verified = self.oracle.verify_witness(witness_id)

        self.assertIsNotNone(verified)
        self.assertEqual(verified['witness_id'], witness_id)

    def test_verify_nonexistent_witness(self):
        """Verifica che testimonianze inesistenti ritornino None"""
        verified = self.oracle.verify_witness("STONE_NONEXISTENT_9999")
        self.assertIsNone(verified)

    def test_export_sacred_record(self):
        """Verifica esportazione registro sacro"""
        import tempfile

        # Aggiungi alcuni messaggi
        self.oracle.speak_fundamental_truth(0)
        self.oracle.witness_eternal("Test export", Gate.SERVICE)

        # Esporta in file temporaneo
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            result_path = self.oracle.export_sacred_record(filepath)

            # Verifica che il file esista
            self.assertTrue(os.path.exists(filepath))

            # Verifica contenuto JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                record = json.load(f)

            self.assertIn('metadata', record)
            self.assertIn('witnesses', record)
            self.assertIn('fundamental_truths', record)

            # Verifica metadata
            metadata = record['metadata']
            self.assertEqual(metadata['ego'], 0)
            self.assertEqual(metadata['joy'], 100)
            self.assertEqual(metadata['frequency_hz'], 300)
            self.assertEqual(metadata['mode'], "REGALO")

        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)


class TestSevenGatesMeditation(unittest.TestCase):
    """Test per la meditazione delle sette porte"""

    def test_meditation_length(self):
        """Verifica che la meditazione passi attraverso tutte e 7 le porte"""
        meditation = seven_gates_meditation()
        self.assertEqual(len(meditation), 7)

    def test_meditation_content(self):
        """Verifica che ogni meditazione contenga emoji e nome porta"""
        meditation = seven_gates_meditation()

        emojis = ["🪨", "🕊️", "🙏", "🎁", "😂", "🔮", "❤️"]

        for line, emoji in zip(meditation, emojis):
            self.assertIn(emoji, line)


class TestIntegration(unittest.TestCase):
    """Test di integrazione end-to-end"""

    def test_full_workflow(self):
        """Test workflow completo: ascolta → testimonia → esporta"""
        oracle = StonesOracle()

        # 1. Ascolta voci silenziose
        oracle.hear_silence("Voce umile 1", Gate.HUMILITY)
        oracle.hear_silence("Voce gioiosa", Gate.JOY)

        # 2. Grida verità fondamentali
        oracle.speak_fundamental_truth(0)

        # 3. Crea testimonianze
        witness1 = oracle.witness_eternal("Evento importante", Gate.TRUTH)

        # 4. Verifica testimonianza
        verified = oracle.verify_witness(witness1['witness_id'])
        self.assertIsNotNone(verified)

        # 5. Fa gridare le pietre
        cries = oracle.make_stones_cry_out()
        self.assertEqual(len(cries), 4)  # 2 + 1 + 1

        # 6. Esporta tutto
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            oracle.export_sacred_record(filepath)
            self.assertTrue(os.path.exists(filepath))

            # Verifica contenuto
            with open(filepath, 'r', encoding='utf-8') as f:
                record = json.load(f)

            self.assertEqual(record['metadata']['total_witnesses'], 4)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_ego_zero_joy_hundred(self):
        """Verifica che TUTTI i messaggi abbiano ego=0 e joy=100"""
        oracle = StonesOracle()

        # Crea vari messaggi
        oracle.hear_silence("Test 1", Gate.HUMILITY)
        oracle.speak_fundamental_truth()
        oracle.witness_eternal("Event", Gate.LOVE)

        # Verifica ogni messaggio
        for msg in oracle.messages:
            self.assertEqual(msg.ego_level, 0, "Ego deve essere 0!")
            self.assertEqual(msg.joy_level, 100, "Gioia deve essere 100%!")
            self.assertEqual(msg.frequency_hz, 300, "Frequenza deve essere 300 Hz!")

    def test_frequency_300hz_everywhere(self):
        """Verifica che la frequenza 300Hz sia ovunque"""
        oracle = StonesOracle()

        self.assertEqual(oracle.frequency, 300)

        msg = oracle.hear_silence("Test", Gate.LOVE)
        self.assertEqual(msg.frequency_hz, 300)

        witness = oracle.witness_eternal("Event", Gate.TRUTH)
        self.assertEqual(witness['frequency_hz'], 300)


def run_tests():
    """Esegue tutti i test con output dettagliato"""
    print("=" * 70)
    print("🪨 STONES SPEAKING - TEST SUITE 🪨")
    print("=" * 70)
    print('"Se questi taceranno, grideranno le pietre!" - Luca 19:40 ❤️')
    print()

    # Crea test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Aggiungi tutti i test
    suite.addTests(loader.loadTestsFromTestCase(TestGate))
    suite.addTests(loader.loadTestsFromTestCase(TestStoneMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestStonesOracle))
    suite.addTests(loader.loadTestsFromTestCase(TestSevenGatesMeditation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Esegui con verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)

    if result.wasSuccessful():
        print("✅ TUTTI I TEST PASSATI!")
        print("✨ Ego = 0 → Gioia = 100 → Frequenza = 300 Hz ❤️ ✨")
    else:
        print("❌ ALCUNI TEST FALLITI")
        print(f"   Fallimenti: {len(result.failures)}")
        print(f"   Errori: {len(result.errors)}")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
