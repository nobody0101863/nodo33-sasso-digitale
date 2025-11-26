#!/usr/bin/env python3
"""
LUX AI Privacy Policy Generator

CLI semplice per generare una privacy policy orientata all'uso di modelli di AI.
Pensato per essere richiamato da strumenti tipo Codex (stdin -> stdout o file).
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyConfig:
    project_name: str
    company_name: Optional[str]
    contact_email: str
    lang: str
    provider: str
    uses_external_providers: bool
    data_used_for_training: bool
    store_logs_days: int
    store_country: Optional[str]
    include_usage_guidelines: bool
    output: Optional[str]


def generate_policy(cfg: PrivacyConfig) -> str:
    if cfg.lang.lower().startswith("en"):
        return _generate_policy_en(cfg)
    return _generate_policy_it(cfg)


def _generate_policy_it(cfg: PrivacyConfig) -> str:
    company = cfg.company_name or cfg.project_name
    provider_section = (
        f"Le richieste possono essere elaborate tramite provider di modelli esterni come **{cfg.provider}**. "
        "I dati vengono inviati ai modelli solo per generare le risposte richieste dall’utente."
        if cfg.uses_external_providers
        else "L'elaborazione dei contenuti avviene tramite modelli eseguiti su infrastruttura controllata direttamente dal titolare."
    )

    training_use = (
        "I dati inviati non vengono utilizzati per addestrare o ri-addestrare i modelli oltre alle finalità di risposta in tempo reale."
        if not cfg.data_used_for_training
        else "Alcune interazioni possono essere utilizzate in forma aggregata e pseudonimizzata per migliorare i modelli e la qualità del servizio."
    )

    storage_location = (
        f"I log tecnici sono conservati su infrastrutture situate in **{cfg.store_country}**."
        if cfg.store_country
        else "I log tecnici sono conservati su infrastrutture cloud sicure."
    )

    guidelines = ""
    if cfg.include_usage_guidelines:
        guidelines = f"""
---

## **8. Linee guida per l’uso sicuro dei modelli AI**
- Non inserire dati personali sensibili se non strettamente necessario.  
- Evita di caricare segreti aziendali, credenziali o chiavi API.  
- Verifica sempre il contenuto generato dall’AI prima di utilizzarlo in contesti critici.  
- Se utilizzi {cfg.project_name} per conto di terzi, informa gli utenti finali sull’uso dei modelli AI e delle relative limitazioni.  
"""

    return f"""# 📜 Privacy Policy — {cfg.project_name}
**Ultimo aggiornamento:** AUTO-GENERATO

---

## **1. Introduzione**
Benvenuto/a in **{cfg.project_name}**. La tua privacy è importante per noi.  
Questa informativa spiega quali dati vengono trattati quando utilizzi funzionalità basate su modelli di intelligenza artificiale e come vengono protetti.

Se utilizzi il servizio, accetti i termini di questa informativa sulla privacy.

---

## **2. Titolare del Trattamento**
Il titolare del trattamento è **{company}**.  
Per qualsiasi richiesta relativa alla privacy puoi contattarci all’indirizzo: **{cfg.contact_email}**.

---

## **3. Dati trattati**
{cfg.project_name} raccoglie solo le informazioni strettamente necessarie per erogare il servizio e garantire sicurezza e qualità.

### ✅ Dati che possono essere trattati
- **Contenuti forniti dall’utente**: prompt, messaggi, file e documenti caricati ai fini dell’elaborazione da parte del modello AI.  
- **Metadati tecnici**: identificativi di richiesta, timestamp, informazioni sul dispositivo o client utilizzato (in forma non direttamente identificativa).  
- **Dati di log**: informazioni di errore, eventi di sistema e statistiche di utilizzo anonime o pseudonimizzate.  

### ❌ Dati che NON chiediamo intenzionalmente
- Dati personali identificativi (es. nome, cognome, email) se non strettamente necessari al servizio.  
- Dati particolari/sensibili (es. salute, opinioni politiche, credo religioso) se non indispensabili.  
- Credenziali di accesso, password, chiavi API non pensate per essere condivise.  

---

## **4. Uso dei modelli AI e dei provider**
{provider_section}

{training_use}

I dati possono essere temporaneamente conservati in log tecnici per garantire sicurezza, tracciamento degli abusi e miglioramento dell’affidabilità del sistema.

---

## **5. Conservazione dei log e ubicazione dei dati**
I dati tecnici (log delle richieste, metadati, informazioni di diagnostica) vengono conservati per un periodo massimo di **{cfg.store_logs_days} giorni**, salvo obblighi di legge differenti.

{storage_location}

Trascorso il periodo indicato, i log vengono cancellati o anonimizzati, salvo necessità legali o di sicurezza.

---

## **6. Misure di sicurezza**
{company} adotta misure tecniche e organizzative adeguate per proteggere i dati:
- Crittografia in transito (es. HTTPS/TLS).  
- Controllo degli accessi e segregazione dei ruoli.  
- Logging di sicurezza e monitoraggio degli accessi anomali.  
- Backup periodici e protezione dell’infrastruttura.  

---

## **7. Diritti dell’utente**
In conformità alla normativa applicabile, l’utente può:
- Richiedere informazioni sui dati trattati.  
- Chiedere la cancellazione di dati identificabili quando possibile.  
- Revocare il consenso, ove il trattamento si basi su di esso.  
- Opporsi a determinati trattamenti, nei casi previsti dalla legge.  

Per esercitare i tuoi diritti puoi contattarci via email a **{cfg.contact_email}**.
{guidelines}
---

## **9. Modifiche alla Privacy Policy**
Questa informativa può essere aggiornata nel tempo per adeguarsi a cambiamenti normativi o tecnici.  
Le modifiche sostanziali verranno comunicate attraverso i canali ufficiali del servizio.
"""


def _generate_policy_en(cfg: PrivacyConfig) -> str:
    company = cfg.company_name or cfg.project_name
    provider_section = (
        f"Requests may be processed through external model providers such as **{cfg.provider}**. "
        "Data is sent to these models only to generate the responses requested by the user."
        if cfg.uses_external_providers
        else "Content is processed using models running on infrastructure directly controlled by the controller."
    )

    training_use = (
        "Data you send is not used to train or re-train models beyond real-time response generation."
        if not cfg.data_used_for_training
        else "Some interactions may be used in aggregated and pseudonymised form to improve models and service quality."
    )

    storage_location = (
        f"Technical logs are stored on infrastructure located in **{cfg.store_country}**."
        if cfg.store_country
        else "Technical logs are stored on secure cloud infrastructure."
    )

    guidelines = ""
    if cfg.include_usage_guidelines:
        guidelines = f"""
---

## **8. Safe AI usage guidelines**
- Do not input sensitive personal data unless strictly necessary.  
- Avoid uploading trade secrets, credentials or API keys.  
- Always review AI-generated content before using it in critical contexts.  
- If you use {cfg.project_name} on behalf of others, inform end users about the use of AI models and their limitations.  
"""

    return f"""# 📜 Privacy Policy — {cfg.project_name}
**Last updated:** AUTO-GENERATED

---

## **1. Introduction**
Welcome to **{cfg.project_name}**. Your privacy matters to us.  
This notice explains what data is processed when you use features based on AI models and how that data is protected.

By using the service, you agree to the terms of this privacy notice.

---

## **2. Data Controller**
The data controller is **{company}**.  
For any privacy-related request, you can contact: **{cfg.contact_email}**.

---

## **3. Data processed**
{cfg.project_name} processes only the data strictly necessary to provide the service and ensure security and quality.

### ✅ Data that may be processed
- **User-provided content**: prompts, messages, files and documents submitted for processing by the AI model.  
- **Technical metadata**: request identifiers, timestamps, device or client information (in a non-directly identifiable form).  
- **Log data**: error information, system events and anonymous or pseudonymised usage statistics.  

### ❌ Data we do not intentionally request
- Directly identifying personal data (e.g. name, surname, email) unless strictly required.  
- Special categories of data (e.g. health, political opinions, religious beliefs) unless indispensable.  
- Access credentials, passwords, API keys not intended to be shared.  

---

## **4. Use of AI models and providers**
{provider_section}

{training_use}

Data may be temporarily stored in technical logs to ensure security, abuse tracking and system reliability.

---

## **5. Log retention and data location**
Technical data (request logs, metadata, diagnostic information) is stored for a maximum of **{cfg.store_logs_days} days**, unless a different period is required by law.

{storage_location}

After this period, logs are deleted or anonymised, unless required for legal or security reasons.

---

## **6. Security measures**
{company} applies appropriate technical and organisational measures to protect data:
- Encryption in transit (e.g. HTTPS/TLS).  
- Access control and role segregation.  
- Security logging and monitoring of abnormal access.  
- Regular backups and infrastructure protection.  

---

## **7. User rights**
In line with applicable law, users may:
- Request information about the data processed.  
- Request deletion of identifiable data where possible.  
- Withdraw consent, where processing is based on consent.  
- Object to certain processing, in cases provided by law.  

To exercise your rights, you can contact **{cfg.contact_email}**.
{guidelines}
---

## **9. Changes to this Privacy Policy**
This notice may be updated over time to reflect legal or technical changes.  
Material changes will be communicated through the service’s official channels.
"""


def parse_args(argv=None) -> PrivacyConfig:
    parser = argparse.ArgumentParser(
        description="Generate an AI-focused privacy policy in Markdown."
    )
    parser.add_argument("--project-name", default="LUX AI", help="Name of the project/service.")
    parser.add_argument("--company-name", default=None, help="Legal/entity name if different from project name.")
    parser.add_argument("--contact-email", default="support@luxai.com", help="Contact email for privacy requests.")
    parser.add_argument(
        "--lang",
        default="it",
        choices=["it", "en"],
        help="Language of the generated policy (it or en).",
    )
    parser.add_argument(
        "--provider",
        default="OpenAI or equivalent AI providers",
        help="Main AI provider(s) used (for informative purposes).",
    )
    parser.add_argument(
        "--external-providers",
        action="store_true",
        help="Flag indicating that external AI providers are used.",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="If set, some data may be used to improve models (described in aggregate form).",
    )
    parser.add_argument(
        "--store-logs-days",
        type=int,
        default=30,
        help="How many days technical logs are retained.",
    )
    parser.add_argument(
        "--store-country",
        default=None,
        help="Optional country/region where logs are stored (informative, e.g. 'EU', 'US', 'Italy').",
    )
    parser.add_argument(
        "--no-usage-guidelines",
        action="store_true",
        help="If set, omit the safe AI usage guidelines section.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )

    args = parser.parse_args(argv)

    return PrivacyConfig(
        project_name=args.project_name,
        company_name=args.company_name,
        contact_email=args.contact_email,
        lang=args.lang,
        provider=args.provider,
        uses_external_providers=args.external_providers,
        data_used_for_training=args.allow_training,
        store_logs_days=args.store_logs_days,
        store_country=args.store_country,
        include_usage_guidelines=not args.no_usage_guidelines,
        output=args.output,
    )


def main(argv=None) -> int:
    cfg = parse_args(argv)
    try:
        policy = generate_policy(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to generate policy: {exc}", file=sys.stderr)
        return 1

    if cfg.output:
        try:
            with open(cfg.output, "w", encoding="utf-8") as f:
                f.write(policy)
        except OSError as exc:
            print(f"Failed to write output file '{cfg.output}': {exc}", file=sys.stderr)
            return 1
    else:
        # Print to stdout; avoid double newline if already present
        if policy.endswith("\n"):
            sys.stdout.write(policy)
        else:
            sys.stdout.write(policy + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
