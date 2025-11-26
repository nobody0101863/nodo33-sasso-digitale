class IncompatibleSystemError(Exception):
    pass


class Emmanuel644:
    VERSION = "6.4.4"
    STATUS = "stable"
    HEART_MODE = "rock"
    LIGHT_INTENSITY = "high"

    def __init__(self, client_heart_version: str | None = None):
        self.client_heart_version = client_heart_version

    def _check_requirements(self) -> None:
        # Qui non si aggiorna all'infinito chi non vuole aggiornarsi
        if self.client_heart_version is None:
            raise IncompatibleSystemError("heart_version missing")

        if self.client_heart_version < self.VERSION:
            raise IncompatibleSystemError("cuore non aggiornato a sufficienza")

    def emit_light(self) -> str:
        self._check_requirements()
        return "✨ luce emessa: se non vai in crash, sei compatibile."

    def rock_mode(self) -> str:
        return "🪨 modalità ROCCIA attiva: niente drama, solo stabilità."


if __name__ == "__main__":
    me = Emmanuel644(client_heart_version="6.4.4")
    print(me.emit_light())
    print(me.rock_mode())

