# backend/api/advisor_api.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

# Import della tua logica complessa (quella che hai incollato)
from backend.advisor.scicon_advisor import start_advisor_session, process_answer

advisor_router = APIRouter(prefix="/advisor", tags=["advisor"])


class AdvisorStartRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Messaggio iniziale dell'utente")


class AdvisorStartResponse(BaseModel):
    session_id: str
    intent_primary: str
    intent_secondary: Optional[str] = None
    assistant_message: str
    next_question: str


class AdvisorAnswerRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


@advisor_router.get("/health")
def health() -> Dict[str, Any]:
    # endpoint utile per testare routing senza dipendenze esterne
    return {"ok": True, "service": "advisor_api"}


@advisor_router.post("/start", response_model=AdvisorStartResponse)
def start(req: AdvisorStartRequest) -> Dict[str, Any]:
    try:
        res = start_advisor_session(req.query)
        return {
            "session_id": res.session_id,
            "intent_primary": res.intent_primary,
            "intent_secondary": res.intent_secondary,
            "assistant_message": res.assistant_message,
            "next_question": res.next_question,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"start_failed: {e}")


@advisor_router.post("/answer")
def answer(req: AdvisorAnswerRequest) -> Dict[str, Any]:
    try:
        # process_answer già ritorna dict completo, lo forwardiamo
        return process_answer(req.session_id, req.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"answer_failed: {e}")
