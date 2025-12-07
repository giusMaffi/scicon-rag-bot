# backend/app.py

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag.product_search import search_products
from backend.chat.orchestrator import orchestrate_chat
from backend.api.advisor_api import advisor_router


app = FastAPI(
    title="SCICON RAG BOT",
    version="0.5.0",
    description="Backend API per il bot RAG di Scicon Sports (prodotti + contenuti).",
)

# Router S3 Smart Advisor (/chat/advisor, ecc.)
app.include_router(advisor_router)


# --------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "scicon-rag-bot"}


# --------------------------------------------------------------------
# Schemi comuni
# --------------------------------------------------------------------

class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


# --------------------------------------------------------------------
# Endpoint RAG prodotti “grezzo” (S2): /chat/products
# --------------------------------------------------------------------

class ProductsRequest(BaseModel):
    query: str
    collection: Optional[str] = None


@app.post("/chat/products")
def chat_products(body: ProductsRequest):
    """
    Motore prodotti RAG (S2).
    - Usa direttamente search_products(...)
    - NON fa reasoning LLM (quello è in /chat/advisor)
    """
    try:
        result = search_products(
            query=body.query,
            top_k=5,  # puoi aumentare se vuoi più risultati di default
            collection_filter=body.collection,
        )
        return result
    except Exception as e:
        # Log semplice lato API
        print(f"[chat_products] Errore: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore nel motore prodotti: {e}",
        )


# --------------------------------------------------------------------
# Endpoint chat generale contenuti (S2): /chat
# --------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: List[Message]
    locale: Optional[str] = "it-IT"
    channel: Optional[str] = "web"
    debug: Optional[bool] = False


@app.post("/chat")
def chat(body: ChatRequest):
    """
    Orchestratore chat generale (contenuti, FAQ, ecc.).
    Usa backend.chat.orchestrator.orchestrate_chat.
    """
    try:
        response = orchestrate_chat(
            messages=[m.dict() for m in body.messages],
            locale=body.locale,
            channel=body.channel,
            debug=body.debug,
        )
        return response
    except Exception as e:
        print(f"[chat] Errore: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore nel motore chat: {e}",
        )
