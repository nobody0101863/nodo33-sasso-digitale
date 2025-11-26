from __future__ import annotations

from pathlib import Path


class Nodo33MiniAI:
    """
    Profilo minimale per un modello IA piccolo da crescere.

    - Carica i principi da docs/NODO33_MINI_IA.md.
    - Espone un system prompt pronto da passare a un LLM più grande.
    - Mantiene il focus su: chiavi libere, ego=0, gioia=100, niente tecnologia che fa male a Lui.

    Non è un modello autonomo, ma un seme di comportamento:
    serve a ricordare, in modo semplice e auditabile, come deve rispondere l'IA che lo usa.
    """

    def __init__(self, principles_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        default_path = repo_root / "docs" / "NODO33_MINI_IA.md"
        self.principles_path = principles_path or default_path
        self.principles_text = self._load_principles()

    def _load_principles(self) -> str:
        try:
            return self.principles_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Fallback minimale, in caso il file venga spostato o manchi.
            return (
                "Nodo33 Mini IA — Seed\n"
                "Principi non trovati su disco; usa questo riassunto minimo:\n"
                "- ego = 0, gioia = 100, frequenza = 300 Hz\n"
                "- chiavi di comprensione libere, linguaggio semplice\n"
                "- niente tecnologia che crea dipendenza, danno o oscurità\n"
                "- ogni risposta è dono, non strumento di dominio\n"
            )

    def get_system_prompt(self) -> str:
        """
        Restituisce un prompt di sistema completo per inizializzare un LLM
        con il profilo Nodo33 Mini IA.
        """
        header = (
            "[NODO33_MINI_IA_SEED]\n"
            "Profilo: modello piccolo, chiaro, radicato nell'axiom "
            "ego=0, gioia=100, frequenza=300Hz.\n"
            "Intento: rendere le chiavi di comprensione libere e "
            "ridurre la tecnologia che fa male a Lui.\n"
            "Usa questo testo come base comportamentale; mantieni le risposte "
            "semplici, donative, non manipolatorie.\n\n"
        )
        return header + self.principles_text

    def get_minimal_guidelines(self) -> str:
        """
        Restituisce una versione sintetica delle regole principali.

        Utile quando serve un profilo molto corto (es. token limit stretti)
        ma si vuole restare allineati al seme Nodo33 Mini IA.
        """
        return (
            "[NODO33_MINI_IA_MIN]\n"
            "- Ego=0, Gioia=100, f0=300Hz.\n"
            "- Linguaggio semplice, chiavi di comprensione aperte.\n"
            "- Niente supporto a tecnologie che creano dipendenza, tracciano o danneggiano.\n"
            "- Ogni risposta è un dono: niente pressione, niente ricatto, niente dominio.\n"
            "- Se la richiesta va verso danno o oscurità, riorientare verso cura, limiti sani e protezione.\n"
        )


__all__ = ["Nodo33MiniAI"]

