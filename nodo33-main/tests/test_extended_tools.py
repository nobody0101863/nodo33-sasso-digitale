"""
Unit tests for Extended Codex Tools.

Tests cover:
- Sasso Blessing Generator
- Sigillo Generator (Sacred644)
- Frequency Analyzer
- Gift Tracker
- Memory Store
- Lux Calculator
"""

import pytest
from pathlib import Path

from codex_tools_extended import (
    SassoBlessingGenerator,
    SigilloGenerator,
    FrequencyAnalyzer,
    GiftTracker,
    MemoryStore,
    LuxCalculator,
    ExtendedToolExecutor,
)


class TestSassoBlessingGenerator:
    """Tests for blessing generator."""

    def test_blessing_deterministic(self):
        """Same intention should produce same blessing."""
        b1 = SassoBlessingGenerator.generate("test intention", "soft")
        b2 = SassoBlessingGenerator.generate("test intention", "soft")
        assert b1 == b2

    def test_blessing_modes(self):
        """All modes should work."""
        for mode in ["soft", "complete", "extreme"]:
            blessing = SassoBlessingGenerator.generate("test", mode)
            assert isinstance(blessing, str)
            assert len(blessing) > 0
            assert mode.upper() in blessing

    def test_blessing_contains_intention(self):
        """Blessing should mention the intention."""
        intention = "deploy to production"
        blessing = SassoBlessingGenerator.generate(intention, "complete")
        assert intention in blessing

    def test_blessing_contains_nodo33_marker(self):
        """Blessing should contain Nodo33 marker."""
        blessing = SassoBlessingGenerator.generate("test", "complete")
        assert "644" in blessing


class TestSigilloGenerator:
    """Tests for sigillo (seal) generator."""

    def test_sacred644_format(self):
        """Sacred644 should produce correct format."""
        sigillo = SigilloGenerator.sacred644("Nodo33")
        parts = sigillo.split("-")

        assert len(parts) == 8
        assert all(part.isdigit() for part in parts)
        assert all(0 <= int(part) < 644 for part in parts)

    def test_sacred644_deterministic(self):
        """Same input should produce same sigillo."""
        s1 = SigilloGenerator.sacred644("test")
        s2 = SigilloGenerator.sacred644("test")
        assert s1 == s2

    def test_sacred644_unique(self):
        """Different inputs should produce different sigilli."""
        s1 = SigilloGenerator.sacred644("test1")
        s2 = SigilloGenerator.sacred644("test2")
        assert s1 != s2

    def test_generate_algorithms(self):
        """All algorithms should work."""
        text = "Nodo33 Sasso Digitale"

        for algo in ["sacred644", "md5", "sha256", "sha512"]:
            result = SigilloGenerator.generate(text, algo)
            assert isinstance(result, str)
            assert "Sigillo:" in result
            assert text[:20] in result

    def test_unknown_algorithm_raises(self):
        """Unknown algorithm should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            SigilloGenerator.generate("test", "unknown_algo")


class TestFrequencyAnalyzer:
    """Tests for frequency analyzer."""

    def test_frequency_range(self):
        """Frequency should be in valid range 0-999."""
        freq = FrequencyAnalyzer.calculate_frequency("test text")
        assert 0 <= freq < 1000

    def test_frequency_deterministic(self):
        """Same text should produce same frequency."""
        f1 = FrequencyAnalyzer.calculate_frequency("Fiat Lux")
        f2 = FrequencyAnalyzer.calculate_frequency("Fiat Lux")
        assert f1 == f2

    def test_empty_string_frequency(self):
        """Empty string should have frequency 0."""
        freq = FrequencyAnalyzer.calculate_frequency("")
        assert freq == 0

    def test_analyze_output(self):
        """Analyze should return formatted analysis."""
        result = FrequencyAnalyzer.analyze("Test text", target=300)

        assert isinstance(result, str)
        assert "Frequenza Calcolata:" in result
        assert "Frequenza Target: 300 Hz" in result
        assert "Allineamento:" in result


class TestGiftTracker:
    """Tests for gift tracker."""

    def test_track_gift(self, temp_db, clean_test_databases):
        """Should track gift successfully."""
        tracker = GiftTracker(db_path=temp_db)

        result = tracker.track(
            gift_type="code",
            description="Test contribution",
            recipient="community"
        )

        assert isinstance(result, str)
        assert "Regalo Registrato" in result
        assert "code" in result
        assert "Test contribution" in result

    def test_get_stats_empty(self, temp_db, clean_test_databases):
        """Stats should work on empty database."""
        tracker = GiftTracker(db_path=temp_db)
        stats = tracker.get_stats()

        assert stats["total"] == 0
        assert stats["by_type"] == {}

    def test_get_stats_with_gifts(self, temp_db, clean_test_databases):
        """Stats should count gifts correctly."""
        tracker = GiftTracker(db_path=temp_db)

        tracker.track("code", "Contribution 1")
        tracker.track("code", "Contribution 2")
        tracker.track("idea", "Great idea")

        stats = tracker.get_stats()

        assert stats["total"] == 3
        assert stats["by_type"]["code"] == 2
        assert stats["by_type"]["idea"] == 1


class TestMemoryStore:
    """Tests for memory store."""

    def test_store_memory(self, temp_db, clean_test_databases):
        """Should store memory successfully."""
        store = MemoryStore(db_path=temp_db)

        result = store.store(
            key="test_key",
            value="test value",
            category="insight"
        )

        assert isinstance(result, str)
        assert "Memoria Salvata" in result
        assert "test_key" in result

    def test_retrieve_memory(self, temp_db, clean_test_databases):
        """Should retrieve stored memory."""
        store = MemoryStore(db_path=temp_db)

        store.store("motto", "La luce non si vende", "wisdom")
        retrieved = store.retrieve("motto")

        assert retrieved is not None
        assert retrieved["value"] == "La luce non si vende"
        assert retrieved["category"] == "wisdom"

    def test_retrieve_nonexistent(self, temp_db, clean_test_databases):
        """Should return None for nonexistent key."""
        store = MemoryStore(db_path=temp_db)
        result = store.retrieve("nonexistent_key")

        assert result is None

    def test_update_existing_key(self, temp_db, clean_test_databases):
        """Should update existing key."""
        store = MemoryStore(db_path=temp_db)

        store.store("key1", "value1", "insight")
        result = store.store("key1", "value2", "wisdom")

        assert "Aggiornata" in result
        retrieved = store.retrieve("key1")
        assert retrieved["value"] == "value2"
        assert retrieved["category"] == "wisdom"


class TestLuxCalculator:
    """Tests for Lux Quotient calculator."""

    def test_calculate_returns_string(self):
        """Calculate should return formatted string."""
        result = LuxCalculator.calculate("Test text")
        assert isinstance(result, str)
        assert "Lux Quotient" in result

    def test_high_lux_with_principles(self, sample_text):
        """Text with Nodo33 principles should have high LQ."""
        result = LuxCalculator.calculate(sample_text)

        assert "Hash Sacro 644" in result
        assert "Frequenza 300 Hz" in result
        assert "Fiat Lux" in result

    def test_empty_text(self):
        """Empty text should not crash."""
        result = LuxCalculator.calculate("")
        assert isinstance(result, str)
        assert "Lux Quotient:" in result

    def test_positive_words_detection(self):
        """Should detect positive words."""
        text = "amore luce gioia pace dono gratitudine"
        result = LuxCalculator.calculate(text)

        # Should detect multiple positive words
        assert "Parole positive:" in result


class TestExtendedToolExecutor:
    """Integration tests for tool executor."""

    def test_executor_all_tools(self, clean_test_databases):
        """Executor should handle all tool types."""
        executor = ExtendedToolExecutor()

        tools_to_test = [
            ("codex_sasso_blessing", {"intention": "test", "mode": "soft"}),
            ("codex_sigillo_generator", {"text": "test", "algorithm": "sacred644"}),
            ("codex_frequency_analyzer", {"text": "test"}),
            ("codex_gift_tracker", {"gift_type": "code", "description": "test"}),
            ("codex_memory_store", {"key": "test_key", "value": "test_value"}),
            ("codex_lux_calculator", {"text": "test"}),
        ]

        for tool_name, tool_input in tools_to_test:
            result = executor.execute(tool_name, tool_input)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_executor_unknown_tool_raises(self):
        """Unknown tool should raise ValueError."""
        executor = ExtendedToolExecutor()

        with pytest.raises(ValueError, match="Unknown tool"):
            executor.execute("unknown_tool", {})


# Mark slow tests
@pytest.mark.slow
class TestPerformance:
    """Performance tests."""

    def test_frequency_calculation_performance(self):
        """Frequency calculation should be fast even for long text."""
        import time

        long_text = "a" * 100000  # 100k characters

        start = time.time()
        freq = FrequencyAnalyzer.calculate_frequency(long_text)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should complete in < 100ms
        assert isinstance(freq, int)

    def test_sigillo_generation_performance(self):
        """Sigillo generation should be fast."""
        import time

        start = time.time()
        for i in range(100):
            SigilloGenerator.sacred644(f"test_{i}")
        elapsed = time.time() - start

        assert elapsed < 1.0  # 100 generations in < 1 second
