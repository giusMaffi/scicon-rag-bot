from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from backend.rag.product_search import search_products
from backend.chat.product_advisor import build_product_advice
from backend.chat.orchestrator import orchestrate_chat

app = FastAPI(
    title="SCICON RAG BOT",
    version="0.5.0",
    description="Backend API per il bot RAG di Scicon Sports (prodotti + contenuti).",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "scicon-rag-bot"}


# ---------- SCHEMI COMUNI ----------

class ProductResult(BaseModel):
    id: Optional[str]
    name: Optional[str]
    url: Optional[str]
    description: Optional[str] = None
    image_url: Optional[str] = None
    sku: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    collection: Optional[str] = None
    features_text: Optional[str] = None
    tech_specs_text: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


# ---------- /products/search ----------

class ProductSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    collection: Optional[str] = None


class ProductSearchResponse(BaseModel):
    results: List[ProductResult]


@app.post("/products/search", response_model=ProductSearchResponse)
def products_search(body: ProductSearchRequest):
    top_k = body.top_k or 5

    results = search_products(
        query=body.query,
        top_k=top_k,
        collection_filter=body.collection,
    )

    return {"results": results}


# ---------- /chat/products (BOT CONTRACT) ----------

class AppliedFilters(BaseModel):
    collection: Optional[str] = None
    price_range: Optional[str] = None
    lens_type: Optional[str] = None


class ProductChatMeta(BaseModel):
    intent: Optional[str] = None
    user_query: Optional[str] = None
    sources: List[str]
    confidence_score: Optional[float] = None
    applied_filters: AppliedFilters
    # campi extra opzionali (es. channel) NON sono tipizzati qui per ora;
    # FastAPI li ignorerà se non sono nel modello, ma restano nel dict lato Python.


class ProductChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    collection: Optional[str] = None


class ProductChatResponse(BaseModel):
    bot_message: str
    products: List[ProductResult]
    follow_up_suggestions: List[str]
    meta: ProductChatMeta


@app.post("/chat/products", response_model=ProductChatResponse)
def chat_products(body: ProductChatRequest):
    top_k = body.top_k or 5

    advice = build_product_advice(
        user_query=body.query,
        top_k=top_k,
        collection_filter=body.collection,
    )

    # advice è già conforme al BOT CONTRACT
    return advice


# ---------- /chat/orchestrator (BOT CONTRACT) ----------

class OrchestratorChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    channel: Optional[str] = None
    collection: Optional[str] = None


# Usiamo lo stesso schema di risposta del bot prodotti,
# perché il BOT CONTRACT è unico.
class OrchestratorChatResponse(ProductChatResponse):
    pass


@app.post("/chat/orchestrator", response_model=OrchestratorChatResponse)
def chat_orchestrator(body: OrchestratorChatRequest):
    top_k = body.top_k or 5

    advice = orchestrate_chat(
        user_query=body.query,
        top_k=top_k,
        channel=body.channel,
        collection=body.collection,
    )

    return advice
