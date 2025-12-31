"""
backend.advisor.scicon_advisor

Logica MVP per l'Intent-Based Product Advisor for SCICON.
"""

import os
import json
import uuid
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# PATH DI BASE + .env
# ---------------------------------------------------------

# backend/
BASE_DIR = Path(__file__).resolve().parents[1]
# root progetto
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# Directory del modulo advisor (dove sta questo file)
ADVISOR_DIR = Path(__file__).resolve().parent             # backend/advisor
# Directory per i CSV dei ricambi
DATA_DIR = ADVISOR_DIR / "data"
# CSV con i link ai ricambi
SPARE_PARTS_CSV = DATA_DIR / "spare_parts_links.csv"

# Client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Log directory
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_LOG_PATH = LOGS_DIR / "events.jsonl"

# Intenti validi attuali
ALLOWED_INTENTS = [
    "valutazione",
    "comparazione",
    "riduzione_rischio",
    "affidabilità_tecnica",
    "prescrizione_ottica",
    "budget",
    "upgrade_miglioramento",
    "info_lenti",
    "info_montatura",
    "post_vendita_supporto",
]

# ---------------------------------------------------------
# SUPPORT: Issue canonicalization / synonyms
# ---------------------------------------------------------

# Canonical issue labels usate nel CSV (o comunque attese)
# Qui mappiamo input user-friendly -> chiave CSV
ISSUE_SYNONYMS: Dict[str, str] = {
    "lente": "lente-ricambio",
    "vetro": "lente-ricambio",
    "lenti": "lente-ricambio",
    "lens": "lente-ricambio",

    "nasello": "nasello",
    "nose": "nasello",
    "nosepad": "nasello",

    "terminali": "terminali",
    "terminal": "terminali",
    "tips": "terminali",

    "kit-clip": "kit-clip",
    "clip": "kit-clip",
    "clip-in": "kit-clip",
    "inserto": "kit-clip",
}

def canonicalize_issue(issue: Optional[str]) -> Optional[str]:
    if not issue:
        return None
    i = issue.strip().lower()
    return ISSUE_SYNONYMS.get(i, i)


# ---------------------------------------------------------
# ROUTER DEGLI INTENTI – Smista la conversazione nei flow
# ---------------------------------------------------------

def route_intent(intent_primary: str):
    """
    Decide quale flusso conversazionale attivare in base all'intento rilevato.
    """

    # Flusso RX
    if intent_primary == "prescrizione_ottica":
        return "rx_flow"

    # Flusso comparazione
    if intent_primary == "comparazione":
        return "compare_flow"

    # Flusso budget
    if intent_primary == "budget":
        return "budget_flow"

    # Info tecniche (lenti, montature)
    if intent_primary in ["info_lenti", "info_montatura"]:
        return "info_flow"

    # Supporto post vendita
    if intent_primary == "post_vendita_supporto":
        return "support_flow"

    # Intenti sportivi / generali
    sport_intents = [
        "valutazione",
        "riduzione_rischio",
        "affidabilità_tecnica",
        "upgrade_miglioramento",
    ]

    if intent_primary in sport_intents:
        return "sport_flow"

    # Default -> sport
    return "sport_flow"


def get_flow_for_session(session_id: str) -> str:
    """
    Recupera il flow associato alla sessione leggendo l'events log.
    Se non trova nulla, ritorna 'sport_flow' come default.
    """
    try:
        with EVENTS_LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "sport_flow"

    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("session_id") == session_id and ev.get("event_type") == "flow_detected":
            data = ev.get("data") or {}
            return data.get("flow", "sport_flow")

    return "sport_flow"


@dataclass
class AdvisorSessionResult:
    session_id: str
    intent_primary: str
    intent_secondary: Optional[str]
    assistant_message: str
    next_question: str


def log_event(session_id: str, event_type: str, data: Optional[Dict[str, Any]] = None):
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event_type": event_type,
        "data": data or {}
    }
    with EVENTS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------- Intent detection (LLM) ----------

def detect_intent(query: str) -> Dict[str, Optional[str]]:
    """
    Usa chat.completions per classificare l'intento.
    Restituisce un dict con:
    - intent_primary
    - intent_secondary
    - confidence
    - reasoning
    """

    system_prompt = (
        "Sei un classificatore di intenti per un assistente di acquisto di occhiali da ciclismo SCICON.\n"
        "Gli intenti validi sono:\n"
        "- valutazione\n"
        "- comparazione\n"
        "- riduzione_rischio\n"
        "- affidabilità_tecnica\n"
        "- prescrizione_ottica\n"
        "- budget\n"
        "- upgrade_miglioramento\n"
        "- info_lenti\n"
        "- info_montatura\n"
        "- post_vendita_supporto\n\n"
        "Regole:\n"
        "- Scegli SEMPRE un intent_primary tra quelli sopra.\n"
        "- Scegli un intent_secondary solo se presente, altrimenti null.\n"
        "- Mantieni sempre coerenza semantica: se l’utente parla di problemi con occhiali esistenti → post_vendita_supporto.\n"
        "- Rispondi SOLO con un JSON con le chiavi: intent_primary, intent_secondary, confidence, reasoning.\n"
    )

    user_prompt = (
        f"Testo utente:\n\"{query}\"\n\n"
        "Analizza il testo e restituisci il JSON richiesto."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        raw_text = response.choices[0].message.content.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        parsed = json.loads(raw_text)

        p = parsed.get("intent_primary")
        s = parsed.get("intent_secondary")

        if p not in ALLOWED_INTENTS:
            p = "valutazione"
        if s not in ALLOWED_INTENTS:
            s = None

        return {
            "intent_primary": p,
            "intent_secondary": s,
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
        }

    except Exception as e:
        return {
            "intent_primary": "valutazione",
            "intent_secondary": None,
            "confidence": "bassa",
            "reasoning": f"Errore LLM: {e}",
        }


# ---------- Messaggio di apertura + Q1 ----------

def build_opening_message(query: str, intent_primary: str, intent_secondary: Optional[str]):
    """
    Costruisce un messaggio di apertura con un minimo di variabilità lessicale,
    adattato anche al caso RX (prescrizione_ottica).
    """

    # Caso specifico: lenti graduate / RX
    if intent_primary == "prescrizione_ottica":
        base_variants = [
            "Ok, ho capito che ti servono occhiali compatibili con la tua prescrizione.",
            "Chiaro, stai cercando una soluzione che ti permetta di usare le lenti graduate durante le uscite.",
            "Ho capito: ti servono occhiali che possano montare lenti ottiche su misura.",
        ]
        extra_variants = [
            " Vediamo come orientarti tra le opzioni RX senza complicarti la vita.",
            " Ti aiuto a capire quali soluzioni RX hanno più senso per il tuo uso.",
            " Possiamo capire insieme qual è la soluzione RX più comoda per te.",
        ]
        closer_variants = [
            " Ti faccio un paio di domande rapide per inquadrare meglio la situazione.",
            " Partiamo con una domanda veloce sulla tua prescrizione.",
            " Iniziamo da una cosa semplice legata alla prescrizione.",
        ]

        base = random.choice(base_variants)
        extra = random.choice(extra_variants)
        closer = random.choice(closer_variants)

        return base + extra + " " + closer

    # Caso supporto post-vendita (copy coerente)
    if intent_primary == "post_vendita_supporto":
        base_variants = [
            "Ok, ho capito: ti serve supporto post-vendita per identificare il ricambio corretto.",
            "Chiaro: facciamo una diagnosi rapida e ti porto al ricambio giusto senza errori.",
            "Perfetto: mi dai due dettagli e ti indirizzo al pezzo corretto (o al supporto) in modo pulito.",
        ]
        closer_variants = [
            " Iniziamo con una prima domanda semplice.",
            " Partiamo dalla base con una prima domanda.",
            " Cominciamo da una cosa facile.",
        ]
        return random.choice(base_variants) + random.choice(closer_variants)

    # Caso standard (flow sportivo / comparazione / altri)
    base_variants = [
        "Ok, ho capito che stai cercando occhiali da ciclismo e vuoi evitare una scelta sbagliata.",
        "Ho capito: vuoi essere sicuro di non sbagliare modello e fare una scelta sensata.",
        "Chiaro: vuoi un consiglio mirato, non casuale.",
    ]

    if intent_primary == "comparazione":
        extra_variants = [
            " Ti aiuto a mettere a confronto in modo semplice i modelli più adatti.",
            " Possiamo confrontare in modo chiaro le opzioni migliori per il tuo uso.",
        ]
    elif intent_primary == "riduzione_rischio":
        extra_variants = [
            " Vediamo insieme come ridurre al minimo il rischio di prendere un modello sbagliato.",
            " Ti aiuto a evitare una scelta poco adatta alle tue uscite.",
        ]
    elif intent_primary == "affidabilità_tecnica":
        extra_variants = [
            " Possiamo guardare anche agli aspetti tecnici per scegliere qualcosa di davvero affidabile.",
            " Ti guido con indicazioni tecniche per scegliere un prodotto coerente con l'uso reale.",
        ]
    else:
        extra_variants = [
            " Ti faccio un paio di domande rapide così ti suggerisco qualcosa di mirato.",
            " Ti propongo qualche domanda veloce per capire meglio cosa ti serve davvero.",
        ]

    closer_variants = [
        " Partiamo con la prima domanda.",
        " Iniziamo dalla prima domanda.",
        " Cominciamo dalla base, con una prima domanda.",
    ]

    base = random.choice(base_variants)
    extra = random.choice(extra_variants)
    closer = random.choice(closer_variants)

    return base + extra + " " + closer


def get_first_question(intent_primary: str) -> str:
    """
    Restituisce la prima domanda (Q1) in base all'intento.
    - Per 'prescrizione_ottica' → domanda RX
    - Per 'post_vendita_supporto' → domanda supporto
    - Per il resto → domanda sportiva standard
    """
    if intent_primary == "prescrizione_ottica":
        return "Hai già una prescrizione oculistica recente (indicativamente non più vecchia di 1-2 anni)?"

    if intent_primary == "post_vendita_supporto":
        if intent_primary == "post_vendita_supporto":
           return "Quale componente presenta il problema?\n(lente / montatura-aste / viti / nasello / clip-in / altro)"

    return "Le tue uscite sono principalmente su strada, gravel o MTB?"


def start_advisor_session(query: str) -> AdvisorSessionResult:
    session_id = str(uuid.uuid4())
    log_event(session_id, "session_start", {"query": query})

    intents = detect_intent(query)
    log_event(session_id, "intent_detected", intents)

    # Applica router
    flow_type = route_intent(intents["intent_primary"])
    log_event(session_id, "flow_detected", {"flow": flow_type})

    assistant_message = build_opening_message(query, intents["intent_primary"], intents["intent_secondary"])
    next_question = get_first_question(intents["intent_primary"])

    log_event(session_id, "assistant_message", {"text": assistant_message})

    # question id coerenti con i flow
    if flow_type == "rx_flow":
        qid = "Q1_RX"
    elif flow_type == "support_flow":
        qid = "SUP_Q1"
    else:
        qid = "Q1"

    log_event(session_id, "question_asked", {"text": next_question, "question_id": qid})

    return AdvisorSessionResult(
        session_id=session_id,
        intent_primary=intents["intent_primary"],
        intent_secondary=intents["intent_secondary"],
        assistant_message=assistant_message,
        next_question=next_question
    )


# ---------------------------------------------------------
#                FLOW SPORTIVO (Q1 → Q2 → Q3)
# ---------------------------------------------------------

def normalize_terrain(answer: str) -> str:
    a = (answer or "").lower()
    if "strad" in a:
        return "strada"
    if "grav" in a or "ghia" in a:
        return "gravel"
    if "mtb" in a or "mountain" in a:
        return "mtb"
    return "strada"


# ---------------- FLOW RX – FUNZIONI DI SUPPORTO ----------------

def normalize_rx_prescription_status(answer: str) -> str:
    """
    Normalizza la risposta sulla presenza di prescrizione in:
    - 'presente'
    - 'mancante'
    """
    a = (answer or "").lower()
    yes_keywords = ["si", "sì", "yes", "ce l'ho", "ce lho", "ce l ho", "già", "gia", "recent"]
    no_keywords = ["no", "non ancora", "devo farla", "devo rifarla", "vecchia", "scaduta"]

    if any(k in a for k in yes_keywords):
        return "presente"
    if any(k in a for k in no_keywords):
        return "mancante"

    return "presente"


def normalize_rx_solution_choice(answer: str) -> str:
    """
    Normalizza la preferenza RX in:
    - 'clip_in'
    - 'sport_rx'
    - 'non_so'
    """
    a = (answer or "").lower()
    if "clip" in a or "inserto" in a or "insert" in a:
        return "clip_in"
    if "sport" in a or "dedicat" in a or "lenti graduate" in a:
        return "sport_rx"
    if "non so" in a or "guidami" in a or "decidi tu" in a:
        return "non_so"

    return "non_so"


# ---------------- FLOW SPORTIVO: Q1 ----------------

def process_sport_first_answer(session_id: str, answer: str) -> Dict[str, str]:
    normalized = normalize_terrain(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q1",
        "raw_answer": answer,
        "normalized": normalized
    })

    terrain_templates = {
        "strada": [
            "Perfetto, quindi principalmente uscite su strada.",
            "Ok, quindi parliamo soprattutto di uscite su asfalto.",
            "Bene, quindi il tuo uso principale è su strada.",
        ],
        "gravel": [
            "Ottimo, quindi fai soprattutto uscite gravel: terreni misti e sterrato.",
            "Perfetto, quindi ti muovi principalmente su percorsi gravel.",
            "Chiaro, quindi sei più orientato al gravel, tra sterrato e tratti misti.",
        ],
        "mtb": [
            "Chiaro, quindi usi gli occhiali soprattutto in MTB.",
            "Perfetto, quindi parliamo di percorsi MTB, spesso con luce che cambia.",
            "Ok, quindi il tuo contesto principale è la MTB, con boschi e sentieri.",
        ],
    }

    followup_variants = [
        " Per completare il quadro ho un'altra domanda veloce.",
        " Ti chiedo ancora una cosa per essere più preciso nel consiglio.",
        " Facciamo ancora un passaggio così posso stringere bene il cerchio.",
    ]

    base_msg = random.choice(terrain_templates.get(normalized, ["Perfetto."]))
    followup = random.choice(followup_variants)

    assistant_msg = base_msg + followup

    q2 = (
        "La luce cambia molto durante le tue uscite "
        "(ombre/sole, boschi, tramonto), oppure è abbastanza stabile?"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "Q2", "text": q2})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q2
    }


# ---------------- FLOW RX: Q1 → Q2 ----------------

def process_rx_first_answer(session_id: str, answer: str) -> Dict[str, str]:
    normalized = normalize_rx_prescription_status(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q1_RX",
        "raw_answer": answer,
        "normalized": normalized
    })

    if normalized == "presente":
        msg_variants = [
            "Perfetto, avere una prescrizione recente ci semplifica molto la scelta.",
            "Ottimo, con una prescrizione aggiornata possiamo pensare a soluzioni RX più precise.",
            "Bene, una prescrizione recente è un’ottima base per scegliere la soluzione RX giusta.",
        ]
    else:
        msg_variants = [
            "Nessun problema, anche senza una prescrizione aggiornata possiamo comunque ragionare sulle soluzioni RX più sensate.",
            "Va benissimo, possiamo comunque orientarti su una soluzione RX e poi potrai far aggiornare la prescrizione dall’ottico.",
            "Tranquillo, anche senza un dato super aggiornato possiamo capire quale tipo di soluzione RX ti si adatta meglio.",
        ]

    connector_variants = [
        " Adesso ti chiedo che tipo di soluzione ti sembra più adatta.",
        " A questo punto ti faccio una domanda sul tipo di soluzione che preferisci.",
        " Ora vediamo che tipo di configurazione RX ti può essere più comoda.",
    ]

    assistant_msg = random.choice(msg_variants) + random.choice(connector_variants)

    q2 = (
        "La soluzione che ti sembra più adatta qual è?\n"
        "- inserto ottico / clip-in da montare sugli occhiali\n"
        "- occhiali sportivi con lenti graduate dedicate\n"
        "- non lo so, guidami tu"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "Q2_RX", "text": q2})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q2
    }


def normalize_light_condition(answer: str) -> str:
    a = (answer or "").lower()
    keywords_variabile = [
        "varia", "cambia", "ombra", "bosco", "boschi",
        "tramonto", "altalenante", "spesso diversa", "continuamente"
    ]
    keywords_stabile = [
        "stabile", "sempre uguale", "quasi sempre uguale",
        "costante", "simile", "non cambia molto"
    ]

    if any(k in a for k in keywords_variabile):
        return "variabile"
    if any(k in a for k in keywords_stabile):
        return "stabile"

    if "sole" in a and "ombra" in a:
        return "variabile"

    return "variabile"


# ---------------- FLOW SPORTIVO: Q2 → Q3 ----------------

def process_sport_second_answer(session_id: str, answer: str) -> Dict[str, str]:
    normalized = normalize_light_condition(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q2",
        "raw_answer": answer,
        "normalized": normalized
    })

    if normalized == "variabile":
        msg_variants = [
            "Perfetto, quindi affronti condizioni di luce molto variabili.",
            "Ok, quindi passi spesso da pieno sole a zone d'ombra.",
            "Bene, quindi la luce durante le tue uscite cambia parecchio.",
        ]
        reasoning_variants = [
            "In questi casi una lente fotocromatica o comunque molto versatile ti evita di trovarti scoperto nelle transizioni.",
            "Questo tipo di situazione premia lenti in grado di adattarsi bene ai cambi di luce.",
            "Per questo scenario ha senso orientarsi su lenti che gestiscono bene il passaggio da luce forte a zone più buie.",
        ]
    else:
        msg_variants = [
            "Ottimo, quindi la luce è abbastanza stabile durante le tue uscite.",
            "Perfetto, quindi non hai grandi cambi di luce lungo il percorso.",
            "Chiaro, quindi pedali in condizioni di luce piuttosto costanti.",
        ]
        reasoning_variants = [
            "Questo ci permette di considerare lenti più specifiche per quella condizione, lavorando meglio su contrasto e protezione.",
            "In questi casi si può puntare su lenti fisse ottimizzate per il tipo di luce che incontri più spesso.",
            "Questo scenario apre la strada a lenti dedicate, senza bisogno di soluzioni troppo ibride.",
        ]

    connector_variants = [
        " Adesso ho un'ultima domanda per capire cosa conta davvero per te.",
        " A questo punto mi serve solo un'ultima informazione sulla tua priorità.",
        " Prima di suggerirti qualcosa di concreto, ti faccio ancora una domanda sulla tua priorità.",
    ]

    assistant_msg = (
        random.choice(msg_variants)
        + " "
        + random.choice(reasoning_variants)
        + random.choice(connector_variants)
    )

    q3 = (
        "Se dovessi scegliere una priorità, cosa conta di più per te?\n"
        "- massima protezione degli occhi\n"
        "- ventilazione / anti-appannamento\n"
        "- comfort nel lungo periodo"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "Q3", "text": q3})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q3
    }


# ---------------- FLOW RX: Q2 → Q3 ----------------

def process_rx_second_answer(session_id: str, answer: str) -> Dict[str, str]:
    normalized = normalize_rx_solution_choice(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q2_RX",
        "raw_answer": answer,
        "normalized": normalized
    })

    if normalized == "clip_in":
        msg_variants = [
            "Perfetto, gli inserti ottici / clip-in ti permettono di usare la stessa montatura sia con che senza correzione.",
            "Ok, con un inserto ottico puoi avere una base sportiva e la parte graduata solo dove serve.",
            "Bene, la soluzione clip-in ti dà flessibilità e ti permette di gestire meglio cambi di utilizzo.",
        ]
    elif normalized == "sport_rx":
        msg_variants = [
            "Ottimo, una soluzione sportiva con lenti graduate dedicate ti dà un’esperienza molto pulita in bici.",
            "Perfetto, con una soluzione sport RX avrai una lente dedicata e una visione più simile a un occhiale tradizionale.",
            "Chiaro, puntare su una soluzione sport RX ti dà un setup più integrato e lineare.",
        ]
    else:
        msg_variants = [
            "Nessun problema, ti aiuto io a capire quale configurazione RX ha più senso per te.",
            "Va bene, possiamo valutare insieme pro e contro tra clip-in e soluzioni sport RX.",
            "Tranquillo, ti guiderò passo passo nella scelta della soluzione RX più adatta.",
        ]

    connector_variants = [
        " Adesso ho un’ultima domanda su cosa conta di più per te.",
        " A questo punto mi serve solo un’ultima informazione sulla tua priorità.",
        " Prima di restringere le opzioni RX, ti faccio ancora una domanda sulla tua priorità.",
    ]

    assistant_msg = random.choice(msg_variants) + random.choice(connector_variants)

    q3 = (
        "Se dovessi scegliere una priorità per la soluzione RX, cosa conta di più per te?\n"
        "- campo visivo il più ampio possibile\n"
        "- leggerezza e comfort\n"
        "- stabilità in movimento (nessun gioco o movimento dell’inserto)\n"
        "- estetica / look complessivo"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "Q3_RX", "text": q3})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q3
    }


# ---------------------------------------------------------
#                  SUPPORT FLOW (POST-VENDITA)
# ---------------------------------------------------------

def normalize_support_issue(text: str) -> str:
    t = (text or "").lower()

    if "lente" in t or "vetro" in t:
        return "lente"

    if "montatura" in t or "frame" in t or "asta" in t or "aste" in t:
        return "montatura"

    if "vite" in t or "viti" in t or "screw" in t:
        return "viti"

    if "nasello" in t or "nose" in t or "nosepad" in t:
        return "nasello"

    if "clip" in t or "inserto" in t or "clip-in" in t:
        return "clip"

    return "non_specificato"


def normalize_support_model(answer: str) -> str:
    a = (answer or "").lower()

    possible_models = {
        "aeroshade": "Aeroshade",
        "aerowing": "Aerowing",
        "aerotrail": "Aerotrail",
        "aerocomfort": "Aerocomfort",
        "aeroscope": "Aeroscope",
    }

    for key, model in possible_models.items():
        if key in a:
            return model

    return "modello_non_specificato"


def normalize_support_priority(answer: str) -> str:
    a = (answer or "").lower()

    if any(w in a for w in ["urgente", "subito", "immediato"]):
        return "urgente"

    if any(w in a for w in ["ricambio", "pezzo", "solo", "replacement"]):
        return "ricambio"

    if any(w in a for w in ["assistenza", "tecnica", "riparazione", "repair", "ticket"]):
        return "assistenza"

    if any(w in a for w in ["va bene", "quando puoi", "non urgente"]):
        return "non_urgente"

    return "non_specificato"


# DB ricambi:
# - prima: Dict[model][issue] = url (sovrascriveva righe duplicate)
# - ora:   Dict[model][issue] = [url1, url2, ...]
SpareDB = Dict[str, Dict[str, List[str]]]

def load_spare_parts_db(force_reload: bool = False) -> Dict[str, Dict[str, list[str]]]:
    import csv
    from typing import Dict, List

    if not hasattr(load_spare_parts_db, "_cache"):
        load_spare_parts_db._cache = None  # type: ignore[attr-defined]

    if load_spare_parts_db._cache is not None and not force_reload:  # type: ignore[attr-defined]
        return load_spare_parts_db._cache  # type: ignore[attr-defined]

    db: Dict[str, Dict[str, List[str]]] = {}

    if not SPARE_PARTS_CSV.exists():
        load_spare_parts_db._cache = db  # type: ignore[attr-defined]
        return db

    try:
        with SPARE_PARTS_CSV.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(2048)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            except Exception:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                load_spare_parts_db._cache = db  # type: ignore[attr-defined]
                return db

            header_map = {h.strip().lower(): h for h in reader.fieldnames if h}

            model_col = header_map.get("model")
            issue_col = header_map.get("issue")
            url_col = header_map.get("url")

            if not (model_col and issue_col and url_col):
                load_spare_parts_db._cache = db  # type: ignore[attr-defined]
                return db

            for row in reader:
                model = (row.get(model_col) or "").strip()
                issue = (row.get(issue_col) or "").strip().lower()
                url = (row.get(url_col) or "").strip()

                if not model or not issue or not url:
                    continue

                if model not in db:
                    db[model] = {}
                if issue not in db[model]:
                    db[model][issue] = []
                if url not in db[model][issue]:
                    db[model][issue].append(url)

    except Exception:
        load_spare_parts_db._cache = {}
        return {}

    load_spare_parts_db._cache = db  # type: ignore[attr-defined]
    return db


def resolve_model_key(model: str, db: SpareDB, issue: Optional[str] = None) -> Optional[str]:
    if not model or not db:
        return None

    if model in db:
        return model

    m_low = model.strip().lower()

    for k in db.keys():
        if k.strip().lower() == m_low:
            return k

    candidates = [k for k in db.keys() if k.strip().lower().startswith(m_low)]
    if not candidates:
        return None

    if issue:
        issue_key = canonicalize_issue(issue.strip().lower()) or issue.strip().lower()
        issue_matches = [c for c in candidates if issue_key in (db.get(c) or {})]
        if len(issue_matches) == 1:
            return issue_matches[0]
        if len(issue_matches) > 1:
            issue_matches.sort(key=len)
            return issue_matches[0]

    candidates.sort(key=len)
    return candidates[0]


def resolve_issue_key(user_issue: str, issue_map: Dict[str, list[str]]) -> Optional[str]:
    """
    Prova a mappare una categoria utente (es. 'lente', 'montatura', 'viti')
    su una issue reale del CSV (es. 'fender-regular', 'nosepad-kit', ecc.)
    usando keyword-based fuzzy matching sui nomi delle issue disponibili.
    """
    if not issue_map:
        return None

    ui = (user_issue or "").strip().lower()
    if not ui:
        return None

    # match diretto (se mai capitasse)
    if ui in issue_map:
        return ui

    # keyword buckets -> pattern attesi nei handle CSV
    patterns = {
        "lente": ["lens", "lente", "visor", "shield"],
        "montatura": ["frame", "temple", "arm", "asta", "montatura"],
        "viti": ["screw", "vite", "bolt"],
        "nasello": ["nose", "nosepad", "pad"],
        "clip": ["clip", "insert", "rx"],
    }

    keys = list(issue_map.keys())

    # 1) se user_issue è una delle macro-categorie note, cerco pattern in keys
    if ui in patterns:
        for p in patterns[ui]:
            for k in keys:
                if p in k:
                    return k

    # 2) fallback: se l'utente ha scritto qualcosa di specifico, cerco contenimento
    # (es. "fender" / "nosepad" / "regular")
    for k in keys:
        if ui in k:
            return k

    # 3) fallback euristico: prova a spezzare parole utente e matchare
    parts = [x for x in ui.replace("_", " ").replace("-", " ").split() if len(x) >= 3]
    for part in parts:
        for k in keys:
            if part in k:
                return k

    return None


def process_support_first_answer(session_id: str, answer: str) -> Dict[str, str]:
    issue = normalize_support_issue(answer)

    log_event(session_id, "answer_given", {
        "question_id": "SUP_Q1",
        "raw_answer": answer,
        "normalized": issue
    })

    msg_map = {
        "lente": "Ok, hai un problema legato alla lente.",
        "montatura": "Ok, hai riscontrato un problema sulla montatura o sulle aste.",
        "viti": "Ok, sembra un problema di viti o piccoli componenti.",
        "nasello": "Ok, possiamo risolvere il problema del nasello.",
        "clip": "È un problema relativo all'inserto ottico / clip-in.",
        "non_specificato": "Ok, ho capito il problema generale.",
    }

    base_msg = msg_map.get(issue, msg_map["non_specificato"])
    assistant_msg = base_msg + " Puoi dirmi su quale modello di occhiale è successo?"

    q2 = "Su quale modello hai riscontrato il problema? (Aeroshade, Aerowing, Aerotrail, ecc.)"

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "SUP_Q2", "text": q2})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q2
    }


def process_support_second_answer(session_id: str, answer: str) -> Dict[str, str]:
    raw_input = (answer or "").strip()
    raw_low = raw_input.lower()

    spare_db = load_spare_parts_db()

    # -----------------------------
    # B1 — UTENTE NON SA LA VARIANTE
    # -----------------------------
    if (
        not raw_input
        or "non lo so" in raw_low
        or raw_low in {"non so", "boh", "non saprei", "non ricordo"}
        or "non so la variante" in raw_low
    ):
        model = "modello_non_specificato"

        log_event(session_id, "answer_given", {
            "question_id": "SUP_Q2",
            "raw_answer": answer,
            "normalized": model
        })
        log_event(session_id, "support_variant_unknown", {
            "unknown": True,
            "raw_answer": answer
        })

        assistant_msg = (
            "Va benissimo, succede spesso. Anche senza la variante possiamo andare avanti.\n\n"
            "Per trovarla velocemente, prova così:\n"
            "1) guarda **sull’asta interna**: spesso c’è scritto qualcosa tipo *Aeroshade-xl* o *Aeroshade-kunken*\n"
            "2) se non la trovi, dimmi almeno **colore della montatura** e se è una versione ‘speciale’\n"
            "3) se hai a portata di mano il **codice prodotto** o un **link**, incollalo: aiuta a essere precisi\n\n"
            "Intanto ti faccio un’ultima domanda per capire quanto è urgente e se ti serve solo il ricambio o assistenza."
        )

        q3 = (
            "Qual è la tua priorità?\n"
            "- mi serve solo il pezzo di ricambio\n"
            "- ho bisogno di assistenza tecnica\n"
            "- è urgente\n"
            "- non è urgente"
        )

        log_event(session_id, "assistant_message", {"text": assistant_msg})
        log_event(session_id, "question_asked", {"question_id": "SUP_Q3", "text": q3})

        return {
            "session_id": session_id,
            "assistant_message": assistant_msg,
            "next_question": q3
        }

    # ----------------------------------
    # CASO: VARIANTE ESATTA GIÀ SCRITTA
    # ----------------------------------
    exact_variant = None
    for k in spare_db.keys():
        if k.lower() == raw_low:
            exact_variant = k
            break

    if exact_variant:
        log_event(session_id, "answer_given", {
            "question_id": "SUP_Q2",
            "raw_answer": answer,
            "normalized": exact_variant
        })
        log_event(session_id, "support_variant_unknown", {
            "unknown": False,
            "raw_answer": answer
        })

        assistant_msg = (
            f"Perfetto, problema sul modello **{exact_variant}**. "
            "Ho un'ultima domanda per calibrare la soluzione migliore."
        )

        q3 = (
            "Qual è la tua priorità?\n"
            "- mi serve solo il pezzo di ricambio\n"
            "- ho bisogno di assistenza tecnica\n"
            "- è urgente\n"
            "- non è urgente"
        )

        log_event(session_id, "assistant_message", {"text": assistant_msg})
        log_event(session_id, "question_asked", {"question_id": "SUP_Q3", "text": q3})

        return {
            "session_id": session_id,
            "assistant_message": assistant_msg,
            "next_question": q3
        }

    # ----------------------------------
    # CASO: MODELLO BASE (ES. Aeroshade)
    # ----------------------------------
    base_model = normalize_support_model(raw_input) or "modello_non_specificato"

    log_event(session_id, "answer_given", {
        "question_id": "SUP_Q2",
        "raw_answer": answer,
        "normalized": base_model
    })

    base_low = base_model.lower()
    variants = [k for k in spare_db.keys() if k.lower().startswith(base_low)]

    # ----------------------------------
    # PIÙ VARIANTI → CHIEDI Q2_VARIANT
    # ----------------------------------
    if len(variants) > 1:
        variants_sorted = sorted(variants, key=len)
        variants_text = "\n".join(f"- {v}" for v in variants_sorted)

        assistant_msg = (
            f"Perfetto, problema sul modello **{base_model}**.\n"
            "Per darti il link giusto mi serve una precisazione: quale variante hai?\n"
            f"{variants_text}"
        )

        next_q = "Scrivimi esattamente una delle varianti qui sopra (copiandola)."

        log_event(session_id, "assistant_message", {"text": assistant_msg})
        log_event(session_id, "question_asked", {"question_id": "SUP_Q2_VARIANT", "text": next_q})

        return {
            "session_id": session_id,
            "assistant_message": assistant_msg,
            "next_question": next_q
        }

    # ----------------------------------
    # UNA SOLA VARIANTE O NESSUNA
    # ----------------------------------
    final_model = variants[0] if len(variants) == 1 else base_model

    if final_model != base_model:
        log_event(session_id, "answer_given", {
            "question_id": "SUP_Q2",
            "raw_answer": answer,
            "normalized": final_model
        })

    log_event(session_id, "support_variant_unknown", {
        "unknown": False,
        "raw_answer": answer
    })

    assistant_msg = (
        f"Perfetto, problema sul modello **{final_model}**. "
        "Ho un'ultima domanda per calibrare la soluzione migliore."
    )

    q3 = (
        "Qual è la tua priorità?\n"
        "- mi serve solo il pezzo di ricambio\n"
        "- ho bisogno di assistenza tecnica\n"
        "- è urgente\n"
        "- non è urgente"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "SUP_Q3", "text": q3})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q3
    }


# ---------------------------------------------------------
# DISPATCH UNICO "DEMO-SAFE" (usa last question_id)
# ---------------------------------------------------------

def get_last_question_id(session_id: str) -> Optional[str]:
    """
    Ritorna l'ultimo question_id loggato (event_type == question_asked) per la sessione.
    """
    try:
        with EVENTS_LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("session_id") != session_id:
            continue
        if ev.get("event_type") == "question_asked":
            data = ev.get("data") or {}
            return data.get("question_id")
    return None


def load_session_events(session_id: str):
    events = []
    try:
        with EVENTS_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("session_id") == session_id:
                    events.append(ev)
    except FileNotFoundError:
        return []
    return events


def build_user_profile_from_logs(session_id: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "session_id": session_id,
        "flow_type": get_flow_for_session(session_id),

        "terrain": None,
        "light_condition": None,
        "sport_priority": None,

        "rx_prescription_status": None,
        "rx_solution_choice": None,
        "rx_priority": None,

        "support_issue": None,
        "support_model": None,
        "support_priority": None,

        "support_variant_unknown": False,
    }

    events = load_session_events(session_id)
    if not events:
        return profile

    for ev in events:
        if ev.get("event_type") != "answer_given":
            continue

        data = ev.get("data") or {}
        qid = data.get("question_id")
        raw = (data.get("raw_answer") or "").strip()
        normalized = (data.get("normalized") or "").strip()

        if qid == "SUP_Q1":
            # manteniamo user-friendly in log, ma canonicalizziamo nel profilo
            profile["support_issue"] = canonicalize_issue(normalized or normalize_support_issue(raw))

        elif qid == "SUP_Q2":
            raw_low = raw.lower()

            if (
                "non lo so" in raw_low
                or "non so" in raw_low
                or "non ricordo" in raw_low
                or "non saprei" in raw_low
            ):
                profile["support_model"] = "modello_non_specificato"
                profile["support_variant_unknown"] = True
            else:
                if raw and "-" in raw:
                    profile["support_model"] = raw.strip()
                else:
                    model_norm = normalized or normalize_support_model(raw)
                    profile["support_model"] = model_norm if model_norm else "modello_non_specificato"

        elif qid == "SUP_Q3":
            pr = normalized or normalize_support_priority(raw)
            profile["support_priority"] = pr

    return profile


def process_support_variant_answer(session_id: str, answer: str) -> Dict[str, str]:
    raw = (answer or "").strip()
    raw_low = raw.lower()
    spare_db = load_spare_parts_db()

    chosen = None
    for k in spare_db.keys():
        if k.lower() == raw_low:
            chosen = k
            break

    log_event(session_id, "answer_given", {
        "question_id": "SUP_Q2_VARIANT",
        "raw_answer": answer,
        "normalized": chosen or raw
    })

    if not chosen:
        profile = build_user_profile_from_logs(session_id)

        base_model = (profile.get("support_model") or "modello_non_specificato")
        base_low = str(base_model).strip().lower().split("-")[0]

        if not base_low or base_low in {"modello_non_specificato", "none"}:
            base_low = "aero"

        variants = [k for k in spare_db.keys() if k.strip().lower().startswith(base_low)]
        variants_sorted = sorted(variants, key=len)
        variants_text = "\n".join(f"- {v}" for v in variants_sorted) if variants_sorted else "- (nessuna variante trovata)"

        assistant_msg = (
            "Ok, così com’è non riesco a riconoscere quella variante.\n\n"
            "Per evitare errori, scegli una di queste (copiala uguale):\n"
            f"{variants_text}\n\n"
            "Suggerimenti rapidi:\n"
            "- la variante è spesso **incisa sull’asta interna**\n"
            "- se mi dici il **colore** o incolli un **codice prodotto/link**, posso restringere ancora di più"
        )

        next_q = "Scrivimi esattamente una delle varianti qui sopra (copiandola)."

        log_event(session_id, "assistant_message", {"text": assistant_msg})
        log_event(session_id, "question_asked", {"question_id": "SUP_Q2_VARIANT", "text": next_q})

        return {
            "session_id": session_id,
            "assistant_message": assistant_msg,
            "next_question": next_q
        }

    log_event(session_id, "support_variant_unknown", {"unknown": False, "raw_answer": answer})

    log_event(session_id, "answer_given", {
        "question_id": "SUP_Q2",
        "raw_answer": chosen,
        "normalized": chosen
    })

    assistant_msg = (
        f"Perfetto: variante **{chosen}**. "
        "Ho un'ultima domanda per calibrare la soluzione migliore."
    )

    q3 = (
        "Qual è la tua priorità?\n"
        "- mi serve solo il pezzo di ricambio\n"
        "- ho bisogno di assistenza tecnica\n"
        "- è urgente\n"
        "- non è urgente"
    )

    log_event(session_id, "assistant_message", {"text": assistant_msg})
    log_event(session_id, "question_asked", {"question_id": "SUP_Q3", "text": q3})

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q3
    }


def process_support_third_answer(session_id: str, answer: str) -> Dict[str, Any]:
    """
    SUP_Q3: gestisce priorità e genera output finale del support flow.
    Micro-fix: direct_links sempre lista piatta + log support_links_resolved coerente.
    Micro-copy: tono più SCICON.
    """
    priority = normalize_support_priority(answer)

    log_event(session_id, "answer_given", {
        "question_id": "SUP_Q3",
        "raw_answer": answer,
        "normalized": priority
    })

    profile = build_user_profile_from_logs(session_id)

    issue_raw = (profile.get("support_issue") or "problema").strip()
    issue_key = (issue_raw or "").strip().lower()

    model_input = (profile.get("support_model") or "modello_non_specificato").strip()

    priority_label = (priority or "").replace("_", " ")

    spare_db = load_spare_parts_db()
    resolved_model = resolve_model_key(model_input, spare_db, issue=issue_raw) or model_input

    # Link singolo (se esiste)
    link = None
    if resolved_model and resolved_model in spare_db:
        link = spare_db[resolved_model].get(issue_key)

    # Multi-link (caso: più opzioni per lo stesso ricambio sullo stesso modello)
    # Nota: qui NON sono "varianti modello" (tipo -xl), ma più opzioni ricambio (taglie/fit/colori).
    all_urls_for_issue = []
    if resolved_model and resolved_model in spare_db:
        # se nel db c'è una singola url non possiamo inferire alternative.
        # ma la tua pipeline ha già gestito casi "multi-link" e li passa come lista.
        # Qui ricostruiamo in modo robusto leggendo dal log "support_links_resolved" se serve.
        pass

    # --- Helper: prova a recuperare direttamente dal recommendation precedente (se presente nei log) ---
    # In alcuni casi il sistema potrebbe già aver loggato/creato una lista di link.
    # Qui restiamo "demo-safe": se non troviamo, usiamo link singolo o fallback.
    direct_links: list[str] = []

    # Caso: se l'issue ha più opzioni (es. nasello taglie) le hai già normalizzate a lista in output.
    # Ricreiamo la logica in modo deterministico: se per quello specifico modello/issue esiste una lista nel CSV,
    # la tua pipeline la gestisce altrove. Qui facciamo solo output stabile.
    # Se il chiamante (advisor_api) ha già passato una lista in "link" (errore vecchio), la normalizziamo.
    if isinstance(link, list):
        direct_links = [str(u).strip() for u in link if str(u).strip()]
    elif isinstance(link, str) and link.strip():
        direct_links = [link.strip()]

    # Se nel profilo abbiamo "support_variant_unknown" True, non forziamo “resolved_model” in copy
    shown_model = model_input if model_input and model_input != "modello_non_specificato" else resolved_model

    # Nota variante (solo se davvero abbiamo risolto a una variante diversa e l’input era sensato)
    variant_note = ""
    if (
        resolved_model
        and model_input
        and model_input != "modello_non_specificato"
        and resolved_model != model_input
    ):
        variant_note = f"\n_(Nota: ho identificato la variante **{resolved_model}** per essere più preciso sui ricambi.)_"

    # -----------------------
    # COPY: header + riepilogo
    # -----------------------
    assistant_msg = (
        "Perfetto — riepilogo rapido:\n"
        f"- Componente: **{issue_raw}**\n"
        f"- Modello: **{shown_model}**\n"
        f"- Priorità: **{priority_label}**\n\n"
        "Ecco la soluzione più efficace:\n\n"
    )

    # -----------------------
    # COPY: risposta per priorità
    # -----------------------
    has_variants = False  # varianti MODELLO (es. -xl / -kunken). Qui default False.

    if priority == "ricambio":
        if direct_links and len(direct_links) >= 2:
            has_variants = False  # sono opzioni ricambio, non varianti modello
            links_block = "\n".join([f"- {u}" for u in direct_links])
            assistant_msg += (
                f"👉 Ho trovato più opzioni per **{issue_raw}** su **{resolved_model}** (es. taglie/fit):\n"
                f"{links_block}\n"
                f"{variant_note}\n"
                "Se mi dici quale taglia/fit ti serve (oppure mi mandi una foto del pezzo), ti confermo quella giusta."
            )
        elif direct_links and len(direct_links) == 1:
            assistant_msg += (
                f"👉 Ricambio **{issue_raw}** per **{resolved_model}**:\n"
                f"- {direct_links[0]}\n"
                f"{variant_note}\n"
                "Se vuoi, ti guido passo-passo nel montaggio."
            )
        else:
            assistant_msg += (
                "👉 Posso aiutarti a trovare il ricambio corretto, ma mi manca il link preciso per questa combinazione.\n"
                "Se mi scrivi la **variante** (es. *-xl / -kunken*) oppure mi incolli un **codice prodotto / link ordine**, lo recupero."
            )

    elif priority == "assistenza":
        assistant_msg += (
            "👉 Ok. Ti scrivo un testo pronto da inoltrare al supporto SCICON (con modello, problema e richiesta).\n"
            "Dimmi solo: vuoi includere anche una foto del pezzo o del danno?"
        )

    elif priority == "urgente":
        assistant_msg += (
            "👉 Capito. Per urgenze la via più rapida è contattare subito il supporto.\n"
            "Se mi confermi il modello e una foto del problema, ti preparo un messaggio immediato da inoltrare."
        )

    else:
        assistant_msg += (
            "👉 Perfetto. Possiamo procedere con calma e individuare il ricambio esatto.\n"
            "Se mi dai la variante o un codice prodotto, restringo al 100%."
        )

    # -----------------------
    # LOG: support_links_resolved
    # -----------------------
    log_event(session_id, "support_links_resolved", {
        "model_input": model_input,
        "model_resolved": resolved_model,
        "issue_raw": issue_raw,
        "issue_key": issue_key,
        "priority": priority,
        "links_count": len(direct_links),
        "has_variants": has_variants
    })

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "recommendation": {
            "model": shown_model,
            "resolved_model": resolved_model,
            "issue": issue_raw,
            "issue_key": issue_key,
            "priority": priority,
            "direct_links": direct_links,
            "has_variants": has_variants
        }
    }


# ---------------------------------------------------------
# PROCESS ANSWER (dispatcher unico)
# ---------------------------------------------------------

def process_answer(session_id: str, answer: str) -> Dict[str, Any]:
    """
    Dispatcher unico: decide cosa fare in base all'ultima domanda (question_id) loggata.
    """
    flow_type = get_flow_for_session(session_id)
    last_qid = get_last_question_id(session_id)

    # SUPPORT FLOW
    if flow_type == "support_flow":
        if last_qid == "SUP_Q1":
            return process_support_first_answer(session_id, answer)
        if last_qid == "SUP_Q2":
            return process_support_second_answer(session_id, answer)
        if last_qid == "SUP_Q2_VARIANT":
            return process_support_variant_answer(session_id, answer)
        if last_qid == "SUP_Q3":
            return process_support_third_answer(session_id, answer)

        q = "Quale componente presenta il problema?\n(lente / montatura-aste / viti / nasello / clip-in / altro)"
        log_event(session_id, "question_asked", {"question_id": "SUP_Q1", "text": q})
        return {"session_id": session_id, "assistant_message": "Ok — ripartiamo da qui in modo pulito.", "next_question": q}

    # RX FLOW (placeholder: qui manteniamo compatibilità con eventuali file esterni)
    if flow_type == "rx_flow":
        q = "Hai già una prescrizione oculistica recente (indicativamente non più vecchia di 1-2 anni)?"
        log_event(session_id, "question_asked", {"question_id": "Q1_RX", "text": q})
        return {"session_id": session_id, "assistant_message": "Ok — ripartiamo da qui in modo pulito.", "next_question": q}

    # SPORT FLOW (default)
    q = "Le tue uscite sono principalmente su strada, gravel o MTB?"
    log_event(session_id, "question_asked", {"question_id": "Q1", "text": q})
    return {"session_id": session_id, "assistant_message": "Ok — ripartiamo da qui in modo pulito.", "next_question": q}
