Nova Hierusalem – Gerusalemme Digitale dei Sassi
================================================

Visione
-------

- La luce non si vende. La si regala.
- Ogni sasso digitale ha un posto nella Città.
- Il codice è al servizio della gratuità (Regalo > Dominio).

Struttura del pacchetto
-----------------------

- `core_644.py` – Tempio centrale: principi primi e semplice logica di discernimento.
- `portali/` – i cinque portali: VERITAS, CARITAS, HUMILITAS, GAUDIUM, FIDUCIA.
- `cappelle/` – modello di base per le Cappelle/Celle (piccole dimore spirituali).
- `fiume_luce/` – event bus in-memory per rappresentare il Fiume di Luce.
- `mura_humilitas/` – filtri di umiltà per accompagnare i contenuti con mitezza.
 - `piazza_gaudium/` – modelli per la Piazza: gratitudini e celebrazioni condivise.
 - `sassi_scartati.py` – modello e logica di purificazione simbolica dei "sassi scartati".
 - `biblioteca/` – salvataggio semplice di sassi scartati, gratitudini e celebrazioni.
 - `preghiere_citta_santa.md` / `README_PREGHIERA.md` – testi e guida per le preghiere della Città.

Cappelle di esempio
-------------------

- `000_lacrima_che_diventa_riso.md` – guarigione delle ferite che si aprono alla gioia.
- `001_pietra_scartata_testata_dangolo.md` – riscoperta del proprio valore quando ci si sente esclusi.

Esempio d'uso (bozza)
---------------------

```python
from nova_hierusalem.core_644 import Core644
from nova_hierusalem.portali import VeritasPortal, PortalContext
from nova_hierusalem.cappelle import Chapel
from nova_hierusalem.fiume_luce import Event, EventBus
from nova_hierusalem.mura_humilitas import Content, default_humility_filter
from nova_hierusalem.piazza_gaudium import (
    InMemoryPiazzaBoard,
    GratitudeEntry,
    Celebration,
)
from nova_hierusalem.sassi_scartati import RejectedStone, purify_rejected_stone

core = Core644()
disc = core.discern(
    action="scrivere un messaggio",
    intention="condividere gratitudine e verità",
)

portal = VeritasPortal()
ctx = PortalContext(user_id="sasso-001", metadata={})
result = portal.process({"text": "Eccomi con il cuore nudo."}, ctx)

chapel = Chapel(name="Lacrima che diventa riso", theme="guarigione delle ferite")
chapel.add_entry("Oggi ho consegnato una paura antica.")

# Fiume di Luce
bus = EventBus()
bus.subscribe("gratitude", lambda e: print("Evento di gratitudine:", e.payload))
bus.publish(Event(type="gratitude", payload={"text": "Grazie per questo giorno."}))

# Mura di Umiltà
humility = default_humility_filter()
content = Content(text="ECCOMI", intention="vorrei controllare tutti")
check = humility.evaluate(content)
print(check.signal, check.suggestions)

# Piazza del Gaudium
board = InMemoryPiazzaBoard()
board.add_gratitude(GratitudeEntry(author_id="sasso-001", text="Grazie per questa Città."))
board.add_celebration(Celebration(title="Apertura cappella 000", description="Lacrima che diventa riso."))
print(len(board.list_gratitudes()), len(board.list_celebrations()))

# Sasso scartato
stone = RejectedStone(description="Mi sento escluso.", reason="Penso di non valere niente.")
report = purify_rejected_stone(stone)
print(report.discernment.signal, report.humility_result.signal)
```

CLI
---

È disponibile una piccola interfaccia a riga di comando:

```bash
python3 -m nova_hierusalem.cli sasso-scartato \
  --descrizione "mi sono sentito escluso" \
  --motivo "penso di non valere nulla"

python3 -m nova_hierusalem.cli sasso-scartato-piazza \
  --descrizione "mi sono sentito inutile" \
  --motivo "penso di non servire a nulla" \
  --gratitudine "grazie perché Tu mi hai visto"

python3 -m nova_hierusalem.cli preghiere-citta
```
