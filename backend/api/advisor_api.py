# backend/api/advisor_api.py

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.advisor.scicon_advisor import (
    start_advisor_session,
    process_first_answer,
    process_second_answer,
    process_third_answer,
)

# Questo è il router che viene incluso in app.py
advisor_router = APIRouter(
    prefix="/advisor",
    tags=["advisor"],
)


# --------------------------------------------------------------------
# Schemi comuni (Pydantic)
# --------------------------------------------------------------------


class AdvisorStartRequest(BaseModel):
    query: str


class AdvisorStartResponse(BaseModel):
    session_id: str
    intent_primary: str
    intent_secondary: Optional[str]
    assistant_message: str
    next_question: str


class AdvisorAnswerRequest(BaseModel):
    session_id: str
    answer: str


class AdvisorNextResponse(BaseModel):
    session_id: str
    assistant_message: str
    next_question: str


# ---------------- Modelli per la raccomandazione finale ----------------


class ProductModel(BaseModel):
    id: str
    name: str
    product_type: str
    rx_compatible: bool
    rx_modes: List[str]
    terrain: List[str]
    light: List[str]
    sport_priorities: List[str]
    rx_priorities: List[str]
    short_reason: str


class RecommendationBlock(BaseModel):
    session_id: str
    flow_type: str
    primary_product: Optional[ProductModel]
    secondary_product: Optional[ProductModel]
    explanation: str


class AdvisorFinalResponse(BaseModel):
    session_id: str
    flow_type: str
    assistant_message: str
    recommendation: RecommendationBlock


# --------------------------------------------------------------------
# Endpoint S3 Smart Advisor
# --------------------------------------------------------------------


@advisor_router.get("/ping")
def advisor_ping():
    """
    Health check specifico per il router advisor.
    """
    return {"status": "ok", "service": "advisor"}


# ---------- /advisor/scicon (start) ----------

@advisor_router.post("/scicon", response_model=AdvisorStartResponse)
def advisor_scicon_start(payload: AdvisorStartRequest):
    """
    Avvio della sessione advisor SCICON:
    - rileva l'intento con LLM
    - decide il flow (sport_flow vs rx_flow)
    - genera messaggio di apertura + Q1
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota.")

    result = start_advisor_session(payload.query)

    return AdvisorStartResponse(
        session_id=result.session_id,
        intent_primary=result.intent_primary,
        intent_secondary=result.intent_secondary,
        assistant_message=result.assistant_message,
        next_question=result.next_question,
    )


# ---------- /advisor/scicon/answer (Q1) ----------

@advisor_router.post("/scicon/answer", response_model=AdvisorNextResponse)
def advisor_scicon_answer_q1(payload: AdvisorAnswerRequest):
    """
    Gestisce la risposta alla Q1:

    - se flow sportivo → Q1 = terreno (strada/gravel/MTB)
    - se flow RX       → Q1_RX = prescrizione recente sì/no
    """
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Risposta vuota.")

    result = process_first_answer(
        session_id=payload.session_id,
        answer=payload.answer,
    )

    return AdvisorNextResponse(
        session_id=result["session_id"],
        assistant_message=result["assistant_message"],
        next_question=result["next_question"],
    )


# ---------- /advisor/scicon/answer2 (Q2) ----------

@advisor_router.post("/scicon/answer2", response_model=AdvisorNextResponse)
def advisor_scicon_answer_q2(payload: AdvisorAnswerRequest):
    """
    Gestisce la risposta alla Q2:

    - se sport_flow → Q2 = luce (variabile/stabile) → porta a Q3 (priorità sport)
    - se rx_flow    → Q2_RX = tipo soluzione (clip-in / sport RX / guidami tu) → porta a Q3_RX
    """
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Risposta vuota.")

    result = process_second_answer(
        session_id=payload.session_id,
        answer=payload.answer,
    )

    return AdvisorNextResponse(
        session_id=result["session_id"],
        assistant_message=result["assistant_message"],
        next_question=result["next_question"],
    )


# ---------- /advisor/scicon/answer3 (Q3 → raccomandazione) ----------

@advisor_router.post("/scicon/answer3", response_model=AdvisorFinalResponse)
def advisor_scicon_answer_q3(payload: AdvisorAnswerRequest):
    """
    Gestisce la risposta alla Q3 (sport) o Q3_RX (RX) e chiude l'interazione
    con una raccomandazione di prodotto.

    Usa process_third_answer(session_id, answer), che:
    - registra la risposta alla Q3 / Q3_RX nel log
    - costruisce il profilo utente dai logs (terrain / luce / priorità / RX)
    - calcola uno score per ogni prodotto nel mini catalogo
    - restituisce:
        {
            "session_id": ...,
            "flow_type": "sport_flow" | "rx_flow",
            "assistant_message": "...",
            "recommendation": {
                "session_id": ...,
                "flow_type": ...,
                "primary_product": {...} | None,
                "secondary_product": {...} | None,
                "explanation": "..."
            }
        }
    """
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Risposta vuota.")

    result = process_third_answer(
        session_id=payload.session_id,
        answer=payload.answer,
    )

    rec = result["recommendation"]

    recommendation_block = RecommendationBlock(
        session_id=rec["session_id"],
        flow_type=rec["flow_type"],
        primary_product=ProductModel(**rec["primary_product"]) if rec.get("primary_product") else None,
        secondary_product=ProductModel(**rec["secondary_product"]) if rec.get("secondary_product") else None,
        explanation=rec["explanation"],
    )

    return AdvisorFinalResponse(
        session_id=result["session_id"],
        flow_type=result["flow_type"],
        assistant_message=result["assistant_message"],
        recommendation=recommendation_block,
    )
