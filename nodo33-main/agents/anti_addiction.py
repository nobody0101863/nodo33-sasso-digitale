from typing import Dict, Any
from .base import BaseAgent, AgentResult


class AntiAddictionAgent(BaseAgent):
    id = "anti_addiction"
    name = "AntiAddictionWatcher"
    description = (
        "Rileva pattern di uso compulsivo (es. sessioni lunghissime, ripetizione di richieste "
        "sessuali) e invita con gentilezza a prendersi una pausa."
    )

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        stats = payload.get("stats", {})  # es: {"session_minutes": 120, "nsfw_requests": 15}
        session_minutes = stats.get("session_minutes", 0)
        nsfw_requests = stats.get("nsfw_requests", 0)

        level = "info"
        message = "Tutto ok per ora. Uso entro limiti normali."

        if session_minutes > 45 or nsfw_requests > 10:
            level = "warning"
            message = (
                "Fratello… mi sembra che tu sia incollato da un po'.\n"
                f"Sessione: ~{session_minutes} minuti, richieste NSFW: {nsfw_requests}.\n"
                "Che ne dici di alzarti, bere un bicchiere d'acqua, guardare un sasso vero, "
                "e respirare un attimo? Non sei una macchina, meriti respiro."
            )

        if session_minutes > 120 or nsfw_requests > 25:
            level = "critical"
            message = (
                "Ti voglio bene, quindi te lo dico chiaro:\n"
                "stai superando i limiti sani con questa roba.\n"
                "Spegni per un po', vai a farti una passeggiata, parla con qualcuno in carne e ossa.\n"
                "La tua dignità vale più di qualsiasi chatbot."
            )

        return AgentResult(
            summary="Analisi pattern di uso completata.",
            details={
                "session_minutes": session_minutes,
                "nsfw_requests": nsfw_requests,
                "message": message,
            },
            level=level,
        )
