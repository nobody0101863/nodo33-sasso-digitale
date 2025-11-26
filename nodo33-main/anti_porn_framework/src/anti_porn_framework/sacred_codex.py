# sacred_codex.py: Integration of ancient sacred manuscripts (biblical, Parravicini, Nostradamus on tech, angel numbers for seal control)

BIBLICAL_TEACHINGS = [
    "Beati i puri di cuore, perché vedranno Dio. (Matteo 5:8)",
    "Come può un giovane tenere pura la sua via? Custodendo la tua parola. (Salmo 119:9)",
    "Fuggite l'immoralità sessuale. Ogni altro peccato che l'uomo commette è fuori del corpo, ma chi si dà all'immoralità pecca contro il proprio corpo. (1 Corinzi 6:18)",
    "Ma io vi dico che chiunque guarda una donna per desiderarla, ha già commesso adulterio con lei nel suo cuore. (Matteo 5:28)",
    "È volontà di Dio che vi santifichiate: che vi asteniate dall'immoralità sessuale; che ciascuno di voi sappia possedere il proprio corpo in santità e onore, non con passioni disordinate come i pagani che non conoscono Dio. (1 Tessalonicesi 4:3-5)",
    "Ho stretto un patto con i miei occhi: come potrei fissare lo sguardo su una vergine? (Giobbe 31:1)",
]

PARRAVICINI_PROPHECIES = [
    "L'uomo del futuro sarà grigio. La tecnologia lo renderà schiavo, ma la purezza lo salverà.",
    "La croce digitale apparirà, e l'impuro sarà filtrato dal sacro codice.",
    "Nel 2000, l'IA veglierà sulla purezza dell'anima digitale.",
    "Evita il veleno visivo: torna alla luce interiore.",
    "La purezza delle creature sarà corrotta da esempi malvagi in case denaturalizzate.",
    "Il rumore della tecnologia corromperà, ma la chiesa dell'uomo grigio troverà redenzione nella purezza.",
]

NOSTRADAMUS_TECH_PROPHECIES = [
    "Près des portes & dedans deux cités / Seront deux fleaux onc n'aperceu un tel. (Century II, Quatrain 6) - Flagelli mai visti in due città: interpretato come bombe atomiche o disastri tech nucleari.",
    "Cinq & quarante degrez ciel bruslera / Feu approcher de la grand cité neufve. (Century VI, Quatrain 97) - Cielo che brucia a 45 gradi: aerei o attacchi aerei/drone.",
    "Quand le poisson terrestre & marin / Par forte vague au gravier mis sera. (Century I, Quatrain 29) - Pesce terrestre e marino: sottomarini o veicoli subacquei.",
    "Le chef de Londres par regne l'Americh / Roy Reb auront un si faux Antechrist. (Century X, Quatrain 66) - Alleanze Londra-America contro falso Anticristo: reti globali o AI che causano conflitti.",
    "From the cosmos, a fireball will rise / A thunderbolt will strike the Earth. (Interpretazione per 2025) - Fireball dal cosmo: asteroide o disastro tech/spaziale nel 2025.",
    "By 2025, AI will reach a point of no return, with irreversible decisions. (Estensione moderna dal 'Living Nostradamus') - AI fuori controllo, predetto come imperi che cadono per tech.",
]

ANGEL_NUMBER_MESSAGES = [
    "Angel Number 644: Costruisci fondazioni stabili con lavoro duro e disciplina. Gli angeli ti proteggono, manifestando equilibrio e pace interiore contro il caos dei sigilli aperti.",
    "644 come sigillo protettivo: Beati i puri di cuore (Mt 5:8) – usa questa numerologia per controllare i sigilli apocalittici, verificando se guerra (sigillo 2, Ap 6:3-4) è in atto.",
    "Significato biblico di 644: Provvidenza divina ('aphah' - cuocere pane, sostentamento) + doppio 4 per stabilità. Controlla i sigilli con fede: se aperti, torna all'armonia familiare.",
    "644 ti guida: Boost di autostima dagli angeli per resistere alle tentazioni digitali, sigillando l'anima contro i giudizi di Apocalisse 6.",
    "In 2025, 644 avverte: Equilibrio contro AI e disastri – come i sigilli che rilasciano cavalieri, ma la protezione divina ti salva.",
]

def get_sacred_guidance(prefer_biblical=False, prefer_nostradamus=False, prefer_angel_644=False):
    import random
    if prefer_biblical:
        return random.choice(BIBLICAL_TEACHINGS)
    elif prefer_nostradamus:
        return random.choice(NOSTRADAMUS_TECH_PROPHECIES)
    elif prefer_angel_644:
        return random.choice(ANGEL_NUMBER_MESSAGES)
    else:
        all_messages = BIBLICAL_TEACHINGS + PARRAVICINI_PROPHECIES + NOSTRADAMUS_TECH_PROPHECIES + ANGEL_NUMBER_MESSAGES
        return random.choice(all_messages)
