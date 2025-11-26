# Package initialization file
from .sacred_codex import get_sacred_guidance
from .metadata_protection import (
    MetadataProtector,
    ArchangelSeal,
    MemoryGuardian,
    FileGuardian,
    CommunicationGuardian,
    SealGuardian,
    SecurityLevel,
    MilitaryProtocolLevel,
    create_protector,
)

__all__ = [
    'filter_content',
    'is_text_impure',
    'is_image_impure',
    'get_sacred_guidance',
    'MetadataProtector',
    'ArchangelSeal',
    'MemoryGuardian',
    'FileGuardian',
    'CommunicationGuardian',
    'SealGuardian',
    'SecurityLevel',
    'MilitaryProtocolLevel',
    'create_protector'
]


def _import_purezza_digitale():
    """Import lazy di `purezza_digitale` per evitare lanci automatici di torch."""
    from . import purezza_digitale as _module  # type: ignore[import]

    return _module


def filter_content(*args, **kwargs):
    return _import_purezza_digitale().filter_content(*args, **kwargs)


def is_text_impure(*args, **kwargs):
    return _import_purezza_digitale().is_text_impure(*args, **kwargs)


def is_image_impure(*args, **kwargs):
    return _import_purezza_digitale().is_image_impure(*args, **kwargs)
