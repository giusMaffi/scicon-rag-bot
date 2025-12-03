"""
backend.chat.product_advisor

Costruisce la risposta "intelligente" per l'endpoint /chat/products:

- chiama search_products(...) per ottenere una lista di Product
- sintetizza i prodotti in un contesto per l'LLM
- chiede all'LLM un consiglio in italiano
- normalizza i prodotti in dict serializzabili per FastAPI/JSON
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from backend.rag.product_search import Product, search_products


# ---------------------------------------------------------------------------
# Caricamento .env dal root del progetto (stessa logica di product_search)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"[product_advisor] Uso .env da: {ENV_PATH}")
else:
    print(f"[product_advisor] ATTENZIONE: .env non trovato in {ENV_PATH}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not OPENAI_API_KEY:
    print("[product_advisor] ⚠️ OPENAI_API_KEY non impostata: userò un messaggio generico senza LLM.")


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def _product_to_dict(p: Product) -> Dict[str, Any]:
    """Converte un oggetto Product in un dict serializzabile in JSON."""
    return {
        "id": p.id,
        "name": p.name,
        "url": p.url,
        "description": p.description,
        "image_url": p.image_url,
        "sku": p.sku,
        "brand": p.brand,
        "price": p.price,
        "currency": p.currency,
        "collection": p.collection,
        "features_text": p.features_text,
        "tech_specs_text": p.tech_specs_text,
        "score": p.score,
        "reason": p.reason,
    }


def _build_products_context(products: List[Product]) -> str:
    """Costruisce un testo riassuntivo dei prodotti per l'LLM."""
    lines: List[str] = []
    for p in products:
        lines.append(
            f"- Nome: {p.name}\n"
            f"  URL: {p.url}\n"
            f"  Prezzo: {p.price} {p.currency or ''}\n"
            f"  Collezione: {p.collection or 'n/d'}\n"
            f"  Brand: {p.brand or 'n/d'}\n"
            f"  Descrizione: {p.description[:300]}...\n"
        )
    return "\n".join(lines)


def _build_fallback_message(user_query: str) -> Dict[str, Any]:
    """Risposta di fallback quando non troviamo prodotti o non possiamo usare l'LLM."""
    bot_message = (
        "Al momento non riesco a trovare un prodotto adatto alla tua richiesta. "
        "Prova a riformulare la domanda indicando:\n"
        "- il tipo di utilizzo (strada, gravel, sterrato, uso misto)\n"
        "- le condizioni di luce (piena, variabile, diffusa, notturna)\n"
        "- eventuali preferenze su lente (fotocromatica, specchiata, più chiara o più scura)."
    )

    return {
        "bot_message": bot_message,
        "products": [],
        "follow_up_suggestions": [
            "Vuoi specificare se usi la bici soprattutto su strada, gravel o sterrato?",
            "Vuoi indicare se preferisci una lente fotocromatica, più chiara o più scura?",
        ],
        "meta": {
            "intent": "product_recommendation",
            "user_query": user_query,
            "sources": ["products_rag"],
            "confidence_score": 0.0,
            "applied_filters": {
                "collection": None,
                "price_range": None,
                "lens_type": None,
            },
        },
    }


# ---------------------------------------------------------------------------
# Funzione principale chiamata dall'orchestrator
# ---------------------------------------------------------------------------

def build_product_advice(
    user_query: str,
    top_k: int = 6,
    collection_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dato un testo libero dell'utente, trova i prodotti più rilevanti e costruisce
    un messaggio di consiglio + payload strutturato.
    """

    # 1) Ricerca prodotti su Qdrant
    products: List[Product] = search_products(
        query=user_query,
        top_k=top_k,
        collection_filter=collection_filter,
    )

    if not products:
        # Nessun prodotto trovato → risposta di fallback
        return _build_fallback_message(user_query)

    # 2) Contesto per l'LLM
    products_context = _build_products_context(products)

    # 3) Costruzione messaggio bot (con o senza LLM)
    if openai_client is None:
        # Nessuna chiave OpenAI: messaggio "statico" costruito sul primo prodotto
        best = products[0]
        bot_message = (
            f"In base alla tua richiesta, ti suggerisco {best.name}.\n\n"
            f"È un modello di {best.brand or 'Scicon Sports'} pensato per un utilizzo versatile. "
            f"Puoi vedere i dettagli qui: {best.url}."
        )
        follow_up_suggestions = [
            "Vuoi che ti suggerisca anche modelli più adatti al pieno sole?",
            "Vuoi filtrare tra modelli più economici o più premium?",
        ]
    else:
        # Usiamo l'LLM per generare un consiglio naturale
        system_msg = (
            "Sei un product advisor di Scicon Sports. "
            "Consigli occhiali da ciclismo in modo chiaro, onesto e concreto, "
            "senza linguaggio promozionale esagerato. Rispondi in italiano."
        )

        user_msg = (
            f"Utente: {user_query}\n\n"
            f"Di seguito hai una lista di prodotti candidati già selezionati (non citarli tutti, "
            f"ma concentrati su quelli più adatti):\n\n"
            f"{products_context}\n\n"
            "Compito:\n"
            "- Suggerisci 1–3 modelli adatti alla richiesta.\n"
            "- Spiega in modo semplice perché li consigli (condizioni di luce, tipo uso, comfort).\n"
            "- Usa un tono pratico, come un commesso competente in un negozio di ciclismo.\n"
        )

        try:
            resp = openai_client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": system_msg,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    },
                ],
            )
            bot_message = resp.output[0].content[0].text
        except Exception as e:
            print(f"[product_advisor] ⚠️ Errore nella chiamata LLM: {e}")
            # Fallback sul primo prodotto
            best = products[0]
            bot_message = (
                f"In base alla tua richiesta, ti suggerisco {best.name}.\n\n"
                f"È un modello di {best.brand or 'Scicon Sports'} pensato per un utilizzo versatile. "
                f"Puoi vedere i dettagli qui: {best.url}."
            )

        follow_up_suggestions = [
            "Vuoi che ti suggerisca anche modelli più adatti al pieno sole?",
            "Preferisci dare priorità al comfort o alla massima protezione della lente?",
            "Ti interessa confrontare questi modelli anche in base al prezzo?",
        ]

    # 4) Confidence score (semplice: punteggio del primo prodotto)
    confidence_score = float(products[0].score) if products and products[0].score is not None else 0.0

    # 5) Normalizzazione prodotti in dict
    products_payload = [_product_to_dict(p) for p in products]

    return {
        "bot_message": bot_message,
        "products": products_payload,
        "follow_up_suggestions": follow_up_suggestions,
        "meta": {
            "intent": "product_recommendation",
            "user_query": user_query,
            "sources": ["products_rag"],
            "confidence_score": confidence_score,
            "applied_filters": {
                "collection": collection_filter,
                "price_range": None,
                "lens_type": None,
            },
        },
    }
