import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server import (
    get_defense_tools,
    get_finance_security_tools,
    get_gis_tools,
    get_infra_tools,
    get_mission_tools,
    get_network_security_tools,
    get_privacy_tools,
    get_torrent_clients,
)


def test_privacy_tools_include_known_category() -> None:
    categories = get_privacy_tools()
    assert any(entry["category"] == "Private Browsing" for entry in categories)


def test_privacy_tools_filter_missing_category() -> None:
    try:
        get_privacy_tools("NonExistent")
    except Exception as exc:  # noqa: PERF203
        assert "Categoria privacy non trovata" in str(exc)


def test_torrent_clients_filtered_by_keyword() -> None:
    results = get_torrent_clients("anonimato")
    assert any("Tribler" in entry["client"] or "I2PSnark" in entry["client"] for entry in results)


def test_torrent_clients_no_focus_returns_all() -> None:
    assert len(get_torrent_clients()) >= 6


def test_mission_tools_include_gmat() -> None:
    results = get_mission_tools()
    assert any(entry["tool"].startswith("GMAT") for entry in results)


def test_mission_tools_filtered_by_focus() -> None:
    results = get_mission_tools("telemetria")
    assert any("Open MCT" in entry["tool"] for entry in results)


def test_defense_tools_include_edr() -> None:
    results = get_defense_tools()
    assert any("Endpoint Detection and Response" in entry["tool"] or "EDR" in entry["tool"] for entry in results)


def test_defense_tools_filtered_focus() -> None:
    results = get_defense_tools("honeypot")
    assert any("Honeypot" in entry["tool"] for entry in results)


def test_defense_tools_include_linux_distros() -> None:
    results = get_defense_tools("linux")
    assert any("Qubes" in entry["tool"] for entry in results)


def test_defense_tools_include_kernel_hardening() -> None:
    results = get_defense_tools("kernel")
    assert any("SELinux" in entry["tool"] or "AppArmor" in entry["tool"] for entry in results)


def test_finance_security_includes_pki() -> None:
    results = get_finance_security_tools("pki")
    assert any("X.509" in entry["tool"] or "TLS" in entry["tool"] for entry in results)


def test_infra_tools_include_kubernetes() -> None:
    results = get_infra_tools("kubernetes")
    assert any("Kubernetes" in entry["tool"] for entry in results)


def test_network_security_tools_include_nmap() -> None:
    results = get_network_security_tools("nmap")
    assert any("Nmap" in entry["tool"] for entry in results)


def test_gis_tools_include_qgis() -> None:
    results = get_gis_tools("qgis")
    assert any("QGIS" in entry["tool"] for entry in results)
