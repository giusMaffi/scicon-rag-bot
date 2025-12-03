"""
backend.chat.product_advisor

Layer "conversazionale" sopra il motore di ricerca prodotti.

- Usa search_products() per trovare i prodotti rilevanti.
- Usa OpenAI per generare una risposta in linguaggio naturale
  che spiega quali modelli consiglia e perché.
- Restituisce un payload strutturato secondo il BOT CONTRACT ufficiale.
"""

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from backend.rag.product_search import search_products

# ---------- CARICAMENTO .ENV ----------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


def _compute_confidence_score(products: List[Dict[str, Any]]) -> float:
    """
    Calcola un confidence_score semplice a partire dagli score RAG dei prodotti.
    Per ora usiamo la media degli score disponibili.
    """
    scores: List[float] = []
    for p in products:
        s = p.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


def _build_follow_up_suggestions(user_query: str, products: List[Dict[str, Any]]) -> List[str]:
    """
    Costruisce alcuni suggerimenti di follow-up generici ma utili.
    In futuro possiamo renderli dinamici in base alla query o ai prodotti.
    """
    suggestions: List[str] = [
        "Vuoi che ti suggerisca anche modelli più adatti al pieno sole?",
        "Preferisci dare priorità al comfort o alla massima protezione della lente?",
        "Ti interessa confrontare questi modelli anche in base al prezzo?"
    ]
    return suggestions


def build_product_advice(
    user_query: str,
    top_k: int = 5,
    collection_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - Esegue una ricerca prodotti.
    - Genera una risposta testuale in italiano che spiega la scelta.
    - Restituisce un dizionario conforme al BOT CONTRACT:

      {
        "bot_message": str,
        "products": [...],
        "follow_up_suggestions": [...],
        "meta": {
            "intent": "product_recommendation",
            "user_query": str,
            "sources": ["products_rag"],
            "confidence_score": float,
            "applied_filters": {
                "collection": str|None,
                "price_range": str|None,
                "lens_type": str|None
            }
        }
      }
    """
    products = search_products(
        query=user_query,
        top_k=top_k,
        collection_filter=collection_filter,
    )

    # Se non troviamo niente, messaggio di fallback
    if not products:
        return {
            "bot_message": (
                "Al momento non riesco a trovare un prodotto adatto alla tua richiesta. "
                "Prova a riformulare la domanda indicando l'uso principale (strada, sterrato, "
                "uso misto) e le condizioni di luce (piena, variabile, diffusa, notturna)."
            ),
            "products": [],
            "follow_up_suggestions": [
                "Vuoi provare a specificare se usi la bici soprattutto su strada o sterrato?",
                "Vuoi indicare se cerchi una lente più chiara, fotocromatica o più scura?"
            ],
            "meta": {
                "intent": "product_recommendation",
                "user_query": user_query,
                "sources": ["products_rag"],
                "confidence_score": 0.0,
                "applied_filters": {
                    "collection": collection_filter,
                    "price_range": None,
                    "lens_type": None,
                },
            },
        }

    client = OpenAI()

    # Costruiamo un contesto compatto da dare al modello
    product_summaries: List[str] = []
    for p in products:
        line = (
            f"- Nome: {p.get('name')}\n"
            f"  SKU: {p.get('sku')}\n"
            f"  Prezzo: {p.get('price')} {p.get('currency')}\n"
            f"  Collezione: {p.get('collection')}\n"
            f"  URL: {p.get('url')}\n"
            f"  Motivazione RAG: {p.get('reason')}\n"
        )
        product_summaries.append(line)

    products_block = "\n".join(product_summaries)

    system_prompt = (
        "Sei un product specialist di Scicon Sports. Rispondi in italiano, con tono chiaro, "
        "amichevole e competente, ma senza essere prolisso.\n\n"
        "Linee guida di risposta:\n"
        "- Rispondi in massimo 2-3 paragrafi brevi.\n"
        "- Presenta sempre 2-3 modelli principali in elenco puntato, non tutta la lista.\n"
        "- Spiega perché sono adatti alla richiesta dell'utente (tipo di lente, condizioni di luce, "
        "tipo di utilizzo: strada, sterrato, uso misto, competizione, training, ecc.).\n"
        "- Non usare la parola 'gravel' a meno che sia stata usata esplicitamente dall'utente.\n"
        "- Se i modelli sono molto simili tra loro (es. stesso modello in colori diversi), "
        "raggruppali e cita le varianti solo come dettaglio.\n"
        "- Chiudi sempre con UNA sola frase di suggerimento pratico o call-to-action "
        "(es. quale modello sceglieresti tu o cosa potrebbe fare l'utente come prossimo passo).\n"
        "- Non inventare informazioni che non emergono dai dati prodotti; usa solo ciò che ti viene fornito."
    )

    user_prompt = (
        f"L'utente ti ha chiesto:\n"
        f"\"{user_query}\"\n\n"
        f"Ecco la lista di prodotti rilevanti da cui scegliere (già filtrata dal motore RAG):\n"
        f"{products_block}\n\n"
        f"Adesso rispondi all'utente seguendo le linee guida."
    )

    chat_resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    bot_message = chat_resp.choices[0].message.content.strip()

    confidence_score = _compute_confidence_score(products)
    follow_ups = _build_follow_up_suggestions(user_query, products)

    return {
        "bot_message": bot_message,
        "products": products,
        "follow_up_suggestions": follow_ups,
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
