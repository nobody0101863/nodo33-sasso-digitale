import pytest
from anti_porn_framework.purezza_digitale import is_text_impure
from anti_porn_framework.sacred_codex import get_sacred_guidance, BIBLICAL_TEACHINGS, NOSTRADAMUS_TECH_PROPHECIES, ANGEL_NUMBER_MESSAGES

def test_text_filter():
    assert is_text_impure("This is porn") == True
    assert is_text_impure("Innocent text") == False

def test_biblical_guidance():
    guidance = get_sacred_guidance(prefer_biblical=True)
    assert guidance in BIBLICAL_TEACHINGS

def test_nostradamus_guidance():
    guidance = get_sacred_guidance(prefer_nostradamus=True)
    assert guidance in NOSTRADAMUS_TECH_PROPHECIES

def test_angel_644_guidance():
    guidance = get_sacred_guidance(prefer_angel_644=True)
    assert guidance in ANGEL_NUMBER_MESSAGES
