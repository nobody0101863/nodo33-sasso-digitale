from typing import Dict, Any
from .base import BaseAgent, AgentResult


class LightAgent(BaseAgent):
    id = "light_agent"
    name = "LightAgent"
    description = (
        "Riceve testo tossico/esplicito e prova a trasmutarlo in qualcosa di sano: "
        "consapevolezza, rispetto, ironia pulita, scelta libera."
    )

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        text = (payload.get("text") or "").strip()

        # Qui puoi in futuro chiamare LLM esterni.
        # Per ora facciamo una risposta di principio.
        reply = (
            "Ok, vedo che il contenuto è parecchio spinto/tossico.\n"
            "Invece di alimentarlo, ti propongo questo:\n\n"
            "- Riconosci che sei umano, non un algoritmo di click.\n"
            "- Il tuo desiderio non è sporco, ma può essere usato contro di te.\n"
            "- Puoi scegliere di non dare tempo, dati e soldi a chi campa su questa debolezza.\n\n"
            "Se vuoi, raccontami cosa ti ha colpito davvero (non i dettagli porno), "
            "ma la sensazione sotto: solitudine? curiosità? noia? rabbia?\n"
            "Da lì possiamo portare un po' di luce, senza moralismi."
        )

        return AgentResult(
            summary="Trasmutazione in luce proposta.",
            details={
                "original_excerpt": text[:200],
                "reply": reply,
            },
            level="info",
        )
