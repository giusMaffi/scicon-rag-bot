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
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# Carica variabili da .env
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

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
    "prescrizione_ottica",  # intento RX
]

# ---------------------------------------------------------
# ROUTER DEGLI INTENTI – Smista la conversazione nei flow
# ---------------------------------------------------------

def route_intent(intent_primary: str):
    """
    Decide quale flusso conversazionale attivare in base all'intento rilevato.
    """

    sport_intents = [
        "valutazione",
        "riduzione_rischio",
        "affidabilità_tecnica",
        "problema_specifico",
    ]

    if intent_primary in sport_intents:
        return "sport_flow"

    if intent_primary == "comparazione":
        return "compare_flow"

    if intent_primary == "budget":
        return "budget_flow"

    if intent_primary == "prescrizione_ottica":
        return "rx_flow"

    info_intents = ["info_lenti", "info_montatura"]
    if intent_primary in info_intents:
        return "info_flow"

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
        f"Gli intenti validi sono: {', '.join(ALLOWED_INTENTS)}.\n\n"
        "Regole:\n"
        "- Scegli SEMPRE un intent_primary.\n"
        "- Scegli un intent_secondary solo se presente, altrimenti null.\n"
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
            "Ok, ho capito che ti servono occhiali SCICON compatibili con la tua prescrizione.",
            "Chiaro, stai cercando una soluzione SCICON che ti permetta di usare le lenti graduate durante le uscite.",
            "Ho capito: ti servono occhiali SCICON che possano montare lenti ottiche su misura.",
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

    # Caso standard (flow sportivo / comparazione / altri)
    base_variants = [
        "Ok, ho capito che stai cercando occhiali da ciclismo per uscite lunghe e vuoi evitare una scelta sbagliata.",
        "Ho capito: ti servono occhiali da ciclismo per uscite lunghe e vuoi essere sicuro di non sbagliare modello.",
        "Chiaro, stai cercando un paio di occhiali da ciclismo per uscite lunghe e vuoi fare una scelta sensata, non casuale.",
    ]

    if intent_primary == "comparazione":
        extra_variants = [
            " Ti aiuto a mettere a confronto in modo semplice i modelli più adatti.",
            " Possiamo confrontare in modo chiaro le opzioni migliori per il tuo uso.",
        ]
    elif intent_primary == "riduzione_rischio":
        extra_variants = [
            " Vediamo insieme come evitare un modello poco adatto alle tue uscite.",
            " Ti aiuto a ridurre al minimo il rischio di prendere un modello sbagliato.",
        ]
    elif intent_primary == "affidabilità_tecnica":
        extra_variants = [
            " Possiamo guardare anche agli aspetti tecnici per trovare qualcosa di davvero affidabile.",
            " Ti guido con indicazioni tecniche per scegliere un prodotto coerente con l'uso che ne farai.",
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
    - Per il resto → domanda sportiva standard
    """
    if intent_primary == "prescrizione_ottica":
        return "Hai già una prescrizione oculistica recente (indicativamente non più vecchia di 1-2 anni)?"

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
    log_event(session_id, "question_asked", {"text": next_question, "question_id": "Q1"})

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
    a = answer.lower()
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
    a = answer.lower()
    yes_keywords = ["si", "sì", "yes", "ce l'ho", "ce lho", "ce l ho", "già", "gia", "recent"]
    no_keywords = ["no", "non ancora", "devo farla", "devo rifarla", "vecchia", "scaduta"]

    if any(k in a for k in yes_keywords):
        return "presente"
    if any(k in a for k in no_keywords):
        return "mancante"

    # default: consideriamo presente per non bloccare il flow
    return "presente"


def normalize_rx_solution_choice(answer: str) -> str:
    """
    Normalizza la preferenza RX in:
    - 'clip_in'
    - 'sport_rx'
    - 'non_so'
    """
    a = answer.lower()
    if "clip" in a or "inserto" in a or "insert" in a:
        return "clip_in"
    if "sport" in a or "dedicat" in a or "lenti graduate" in a:
        return "sport_rx"
    if "non so" in a or "guidami" in a or "decidi tu" in a:
        return "non_so"

    return "non_so"


# ---------------- FLOW SPORTIVO: Q1 ----------------

def process_sport_first_answer(session_id: str, answer: str) -> Dict[str, str]:
    """
    Gestione della risposta alla Q1 del flow sportivo.
    """
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
    """
    Gestisce la risposta alla Q1 RX:
    - Hai già una prescrizione recente?
    Genera Q2 RX sulla tipologia di soluzione (clip-in vs sport RX).
    """
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
    log_event(session_id, "question_asked", {
        "question_id": "Q2_RX",
        "text": q2
    })

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q2
    }


def normalize_light_condition(answer: str) -> str:
    a = answer.lower()
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
    """
    Gestione della risposta alla Q2 del flow sportivo.
    """
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
    """
    Gestisce la risposta alla Q2 RX:
    - preferenza tra clip-in / sport RX / non so
    Genera Q3 RX sulla priorità (campo visivo / leggerezza / stabilità / estetica).
    """
    normalized = normalize_rx_solution_choice(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q2_RX",
        "raw_answer": answer,
        "normalized": normalized
    })

    if normalized == "clip_in":
        msg_variants = [
            "Perfetto, gli inserti ottici / clip-in ti permettono di usare la stessa montatura sia con che senza correzione.",
            "Ok, con un inserto ottico puoi avere una base sportiva SCICON e la parte graduata solo dove serve.",
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
    log_event(session_id, "question_asked", {
        "question_id": "Q3_RX",
        "text": q3
    })

    return {
        "session_id": session_id,
        "assistant_message": assistant_msg,
        "next_question": q3
    }


# ---------------------------------------------------------
#        DISPATCH GENERICO PER Q1 E Q2 (SPORT vs RX)
# ---------------------------------------------------------

def process_first_answer(session_id: str, answer: str) -> Dict[str, str]:
    """
    Dispatch generico:
    - Se la session è sport_flow → usa process_sport_first_answer
    - Se la session è rx_flow    → usa process_rx_first_answer
    """
    flow_type = get_flow_for_session(session_id)

    if flow_type == "rx_flow":
        return process_rx_first_answer(session_id, answer)

    # default: flow sportivo
    return process_sport_first_answer(session_id, answer)


def process_second_answer(session_id: str, answer: str) -> Dict[str, str]:
    """
    Dispatch generico:
    - Se la session è sport_flow → usa process_sport_second_answer
    - Se la session è rx_flow    → usa process_rx_second_answer
    """
    flow_type = get_flow_for_session(session_id)

    if flow_type == "rx_flow":
        return process_rx_second_answer(session_id, answer)

    return process_sport_second_answer(session_id, answer)


# ---------------------------------------------------------
#       UTILS: PROFILO UTENTE DA LOGS (Q1–Q3 / RX)
# ---------------------------------------------------------

def normalize_sport_priority(answer: str) -> Optional[str]:
    """
    Normalizza la priorità sportiva (Q3) in:
    - 'protezione'
    - 'ventilazione'
    - 'comfort'
    """
    if not answer:
        return None
    a = answer.lower()
    if "protez" in a or "occhi" in a or "sicurezz" in a:
        return "protezione"
    if "ventil" in a or "appann" in a:
        return "ventilazione"
    if "comfort" in a or "lungo" in a or "ore" in a:
        return "comfort"
    return None


def normalize_rx_priority(answer: str) -> Optional[str]:
    """
    Normalizza la priorità RX (Q3_RX) in:
    - 'campo_visivo'
    - 'comfort'
    - 'stabilita'
    - 'estetica'
    """
    if not answer:
        return None
    a = answer.lower()
    if "campo" in a or "ampio" in a or "visiv" in a:
        return "campo_visivo"
    if "leggerezza" in a or "leggero" in a or "comfort" in a:
        return "comfort"
    if "stabil" in a or "muove" in a or "gioco" in a or "inserto" in a:
        return "stabilita"
    if "estetica" in a or "look" in a or "stile" in a:
        return "estetica"
    return None


def load_session_events(session_id: str):
    """
    Restituisce tutti gli eventi di una sessione come lista (in ordine cronologico).
    """
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
    """
    Ricostruisce un profilo utente leggendo gli answer_given nel log.
    Output di esempio:
    {
        "session_id": "...",
        "flow_type": "sport_flow" | "rx_flow",
        "terrain": "strada" | "gravel" | "mtb" | None,
        "light_condition": "variabile" | "stabile" | None,
        "sport_priority": "protezione" | "ventilazione" | "comfort" | None,
        "rx_prescription_status": "presente" | "mancante" | None,
        "rx_solution_choice": "clip_in" | "sport_rx" | "non_so" | None,
        "rx_priority": "campo_visivo" | "comfort" | "stabilita" | "estetica" | None,
    }
    """
    profile: Dict[str, Any] = {
        "session_id": session_id,
        "flow_type": get_flow_for_session(session_id),
        "terrain": None,
        "light_condition": None,
        "sport_priority": None,
        "rx_prescription_status": None,
        "rx_solution_choice": None,
        "rx_priority": None,
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
        normalized = (data.get("normalized") or "").strip() or None

        if qid == "Q1":
            profile["terrain"] = normalized or normalize_terrain(raw)
        elif qid == "Q2":
            profile["light_condition"] = normalized or normalize_light_condition(raw)
        elif qid == "Q3":
            profile["sport_priority"] = normalized or normalize_sport_priority(raw)
        elif qid == "Q1_RX":
            profile["rx_prescription_status"] = normalized or normalize_rx_prescription_status(raw)
        elif qid == "Q2_RX":
            profile["rx_solution_choice"] = normalized or normalize_rx_solution_choice(raw)
        elif qid == "Q3_RX":
            profile["rx_priority"] = normalized or normalize_rx_priority(raw)

    return profile


# ---------------------------------------------------------
#       MINI CATALOGO PRODOTTI + LOGICA DI SCORING
# ---------------------------------------------------------

# Catalogo fittizio ma coerente con il brand SCICON.
# Non è collegato al RAG, serve solo per il prototipo di raccomandazione.

PRODUCT_CATALOG = [
    {
        "id": "aerotrail_photo",
        "name": "SCICON Aerotrail Photochromic",
        "product_type": "sport",
        "rx_compatible": True,
        "rx_modes": ["clip_in"],
        "terrain": ["strada", "gravel"],
        "light": ["variabile"],
        "sport_priorities": ["protezione", "ventilazione", "comfort"],
        "rx_priorities": [],
        "short_reason": "lente fotocromatica molto versatile, montatura leggera e buona ventilazione.",
    },
    {
        "id": "aeroshade",
        "name": "SCICON Aeroshade Performance",
        "product_type": "sport",
        "rx_compatible": True,
        "rx_modes": ["clip_in"],
        "terrain": ["strada", "gravel"],
        "light": ["stabile", "variabile"],
        "sport_priorities": ["protezione", "comfort"],
        "rx_priorities": [],
        "short_reason": "schermo ampio, copertura massima e look molto racing.",
    },
    {
        "id": "aerowing",
        "name": "SCICON Aerowing",
        "product_type": "sport",
        "rx_compatible": False,
        "rx_modes": [],
        "terrain": ["strada"],
        "light": ["stabile"],
        "sport_priorities": ["protezione", "comfort"],
        "rx_priorities": [],
        "short_reason": "occhiale iconico, ottima protezione e forte identità estetica.",
    },
    {
        "id": "aero_rx_clip",
        "name": "SCICON Aeroshade + Clip-in RX",
        "product_type": "rx",
        "rx_compatible": True,
        "rx_modes": ["clip_in"],
        "terrain": ["strada", "gravel"],
        "light": ["variabile", "stabile"],
        "sport_priorities": ["protezione", "comfort"],
        "rx_priorities": ["campo_visivo", "stabilita"],
        "short_reason": "base performance Aeroshade con inserto ottico stabile e campo visivo ampio.",
    },
    {
        "id": "aero_rx_sport",
        "name": "SCICON Sport RX dedicato",
        "product_type": "rx",
        "rx_compatible": True,
        "rx_modes": ["sport_rx"],
        "terrain": ["strada"],
        "light": ["stabile", "variabile"],
        "sport_priorities": ["comfort", "protezione"],
        "rx_priorities": ["comfort", "estetica", "campo_visivo"],
        "short_reason": "montatura pensata per lenti graduate dedicate, molto pulita e vicina a un occhiale tradizionale.",
    },
]


def score_product_for_sport(profile: Dict[str, Any], product: Dict[str, Any]) -> int:
    """
    Calcola uno score semplice per i flow sportivi.
    """
    score = 0

    terrain = profile.get("terrain")
    if terrain and terrain in product.get("terrain", []):
        score += 3

    light = profile.get("light_condition")
    if light and light in product.get("light", []):
        score += 3

    priority = profile.get("sport_priority")
    if priority and priority in product.get("sport_priorities", []):
        score += 4

    # Piccolo bonus se è un prodotto sport puro
    if product.get("product_type") == "sport":
        score += 1

    return score


def score_product_for_rx(profile: Dict[str, Any], product: Dict[str, Any]) -> int:
    """
    Calcola uno score per i flow RX.
    Considera:
    - tipo di configurazione (clip_in vs sport_rx)
    - priorità RX
    - compatibilità RX del prodotto
    """
    score = 0

    if not product.get("rx_compatible"):
        return 0

    rx_choice = profile.get("rx_solution_choice")  # clip_in | sport_rx | non_so
    if rx_choice == "clip_in" and "clip_in" in product.get("rx_modes", []):
        score += 4
    elif rx_choice == "sport_rx" and "sport_rx" in product.get("rx_modes", []):
        score += 4
    elif rx_choice == "non_so" and product.get("rx_modes"):
        score += 2  # generico, va bene qualsiasi cosa RX-ready

    rx_priority = profile.get("rx_priority")
    if rx_priority and rx_priority in product.get("rx_priorities", []):
        score += 4

    # Usiamo anche terreno/luminosità se presenti
    terrain = profile.get("terrain")
    if terrain and terrain in product.get("terrain", []):
        score += 2

    light = profile.get("light_condition")
    if light and light in product.get("light", []):
        score += 2

    # Piccolo bonus se è chiaramente un prodotto RX
    if product.get("product_type") == "rx":
        score += 1

    return score


def pick_top_products(scored_products):
    """
    Dato un elenco di tuple (product, score), restituisce:
    - primary_product
    - secondary_product (se disponibile)
    """
    if not scored_products:
        return None, None

    scored_products = sorted(scored_products, key=lambda x: x[1], reverse=True)
    primary = scored_products[0][0]
    secondary = scored_products[1][0] if len(scored_products) > 1 and scored_products[1][1] > 0 else None
    return primary, secondary


def build_explanation(profile: Dict[str, Any],
                      primary: Dict[str, Any],
                      secondary: Optional[Dict[str, Any]]) -> str:
    """
    Costruisce una motivazione in linguaggio naturale basata su profilo e prodotti.
    """
    flow_type = profile.get("flow_type")
    pieces = []

    if flow_type == "rx_flow":
        intro = "Ti suggerisco questa configurazione RX partendo da quello che mi hai detto."
    else:
        intro = "Ti suggerisco questi occhiali partendo da come usi la bici e da ciò che per te è prioritario."
    pieces.append(intro)

    # Profilo sintetico
    if profile.get("terrain"):
        pieces.append(f" Usi gli occhiali principalmente su {profile['terrain']}.")
    if profile.get("light_condition"):
        if profile["light_condition"] == "variabile":
            pieces.append(" Affronti spesso condizioni di luce variabile.")
        else:
            pieces.append(" Pedali di solito in condizioni di luce abbastanza stabili.")
    if flow_type != "rx_flow" and profile.get("sport_priority"):
        mapping = {
            "protezione": "massima protezione degli occhi",
            "ventilazione": "buona ventilazione e anti-appannamento",
            "comfort": "comfort nel lungo periodo",
        }
        desc = mapping.get(profile["sport_priority"], profile["sport_priority"])
        pieces.append(f" Hai indicato come priorità {desc}.")
    if flow_type == "rx_flow":
        if profile.get("rx_prescription_status"):
            if profile["rx_prescription_status"] == "presente":
                pieces.append(" Hai già una prescrizione recente.")
            elif profile["rx_prescription_status"] == "mancante":
                pieces.append(" Non hai ancora una prescrizione aggiornata.")
        if profile.get("rx_solution_choice"):
            mapping_choice = {
                "clip_in": "un inserto ottico / clip-in",
                "sport_rx": "una soluzione sportiva con lenti graduate dedicate",
                "non_so": "una soluzione RX da definire insieme",
            }
            pieces.append(f" Come configurazione ti orienti verso {mapping_choice.get(profile['rx_solution_choice'], 'una soluzione RX flessibile')}.")
        if profile.get("rx_priority"):
            mapping_rx = {
                "campo_visivo": "campo visivo ampio",
                "comfort": "leggerezza e comfort",
                "stabilita": "stabilità in movimento dell'inserto",
                "estetica": "estetica e look complessivo",
            }
            pieces.append(f" Per te conta soprattutto {mapping_rx.get(profile['rx_priority'], profile['rx_priority'])}.")

    # Spiegazione legata ai prodotti
    pieces.append(
        f" Per questo ti propongo come prima scelta **{primary['name']}**, "
        f"perché {primary['short_reason']}"
    )

    if secondary:
        pieces.append(
            f" Come seconda opzione puoi considerare **{secondary['name']}**, "
            f"che rimane coerente con le tue esigenze ma con un'impostazione leggermente diversa."
        )

    return " ".join(pieces)


def recommend_products_for_session(session_id: str) -> Dict[str, Any]:
    """
    Entry point principale per il motore di raccomandazione MVP.

    Usa:
    - build_user_profile_from_logs(session_id)
    - PRODUCT_CATALOG
    per restituire:
    {
        "session_id": ...,
        "flow_type": "sport_flow" | "rx_flow",
        "primary_product": {...} | None,
        "secondary_product": {...} | None,
        "explanation": "..."
    }
    """
    profile = build_user_profile_from_logs(session_id)
    flow_type = profile.get("flow_type", "sport_flow")

    scored = []

    if flow_type == "rx_flow":
        for product in PRODUCT_CATALOG:
            s = score_product_for_rx(profile, product)
            if s > 0:
                scored.append((product, s))
        # fallback se nessun prodotto ha score > 0
        if not scored:
            for product in PRODUCT_CATALOG:
                if product.get("product_type") == "rx":
                    scored.append((product, 1))
    else:
        for product in PRODUCT_CATALOG:
            s = score_product_for_sport(profile, product)
            if s > 0:
                scored.append((product, s))
        if not scored:
            for product in PRODUCT_CATALOG:
                if product.get("product_type") == "sport":
                    scored.append((product, 1))

    primary, secondary = pick_top_products(scored) if scored else (None, None)

    if primary:
        explanation = build_explanation(profile, primary, secondary)
    else:
        explanation = (
            "Non ho abbastanza informazioni per suggerirti un modello preciso, "
            "ma possiamo affinare il consiglio con qualche domanda in più."
        )

    return {
        "session_id": session_id,
        "flow_type": flow_type,
        "primary_product": primary,
        "secondary_product": secondary,
        "explanation": explanation,
    }


# ---------------------------------------------------------
#       Q3 / Q3_RX → RACCOMANDAZIONE FINALE
# ---------------------------------------------------------

def process_sport_third_answer(session_id: str, answer: str) -> Dict[str, Any]:
    """
    Gestisce la risposta alla Q3 del flow sportivo e chiude con la raccomandazione.
    """
    normalized = normalize_sport_priority(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q3",
        "raw_answer": answer,
        "normalized": normalized
    })

    rec = recommend_products_for_session(session_id)
    primary = rec.get("primary_product")
    secondary = rec.get("secondary_product")
    explanation = rec.get("explanation")

    # Messaggio finale per l'utente
    lines = [explanation]

    if primary:
        lines.append(f"\n\n👉 Scelta principale: **{primary['name']}**")
    if secondary:
        lines.append(f"\n➕ Seconda opzione: **{secondary['name']}**")

    lines.append("\n\nSe vuoi, possiamo affinare ancora il consiglio confrontando questi modelli o aggiungendo il budget.")

    assistant_msg = "".join(lines)

    log_event(session_id, "recommendation_generated", {
        "flow_type": "sport_flow",
        "primary_product": primary,
        "secondary_product": secondary,
        "explanation": explanation,
    })

    return {
        "session_id": session_id,
        "flow_type": "sport_flow",
        "assistant_message": assistant_msg,
        "recommendation": rec,
    }


def process_rx_third_answer(session_id: str, answer: str) -> Dict[str, Any]:
    """
    Gestisce la risposta alla Q3_RX del flow RX e chiude con la raccomandazione.
    """
    normalized = normalize_rx_priority(answer)

    log_event(session_id, "answer_given", {
        "question_id": "Q3_RX",
        "raw_answer": answer,
        "normalized": normalized
    })

    rec = recommend_products_for_session(session_id)
    primary = rec.get("primary_product")
    secondary = rec.get("secondary_product")
    explanation = rec.get("explanation")

    lines = [explanation]

    if primary:
        lines.append(f"\n\n👉 Configurazione principale: **{primary['name']}**")
    if secondary:
        lines.append(f"\n➕ Seconda opzione RX: **{secondary['name']}**")

    lines.append("\n\nSe vuoi, possiamo entrare più nel dettaglio su lenti, spessori o alternative di montatura.")

    assistant_msg = "".join(lines)

    log_event(session_id, "recommendation_generated", {
        "flow_type": "rx_flow",
        "primary_product": primary,
        "secondary_product": secondary,
        "explanation": explanation,
    })

    return {
        "session_id": session_id,
        "flow_type": "rx_flow",
        "assistant_message": assistant_msg,
        "recommendation": rec,
    }


def process_third_answer(session_id: str, answer: str) -> Dict[str, Any]:
    """
    Dispatch generico per la terza risposta:
    - Se la session è sport_flow → process_sport_third_answer
    - Se la session è rx_flow    → process_rx_third_answer
    """
    flow_type = get_flow_for_session(session_id)

    if flow_type == "rx_flow":
        return process_rx_third_answer(session_id, answer)

    return process_sport_third_answer(session_id, answer)
