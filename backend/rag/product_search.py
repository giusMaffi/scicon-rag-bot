"""
backend.rag.product_search

Funzioni di ricerca prodotti su Qdrant per il bot Scicon.

- Usa un vettore di embedding (OpenAI) per la query utente
- Esegue una ricerca semantica su Qdrant
- Restituisce una lista di prodotti normalizzati con punteggio e reason

Il file è stato reso resiliente:
- Se manca OPENAI_API_KEY o QDRANT_URL / QDRANT_API_KEY, non crasha:
  - logga un messaggio
  - ritorna lista vuota [] da search_products
- Gestisce timeout / errori Qdrant con try/except.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ─────────────────────────────────────────────
# Caricamento .env dalla root del progetto
# ─────────────────────────────────────────────

# Percorso: backend/rag/product_search.py -> backend/rag -> backend -> scicon-rag-bot
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"[product_search] Uso .env da: {ENV_PATH}")
else:
    print(
        f"[product_search] Nessun .env trovato in {ENV_PATH}. "
        "Uso solo le variabili di ambiente già presenti."
    )

# ─────────────────────────────────────────────
# Variabili di ambiente
# ─────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scicon_products")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print(
        "[product_search] OPENAI_API_KEY non impostata: "
        "il calcolo degli embedding/LLM potrebbe fallire."
    )

if not QDRANT_URL or not QDRANT_API_KEY:
    print(
        "[product_search] QDRANT_URL o QDRANT_API_KEY non impostati: "
        "la ricerca prodotti su Qdrant potrebbe fallire."
    )

# ─────────────────────────────────────────────
# Client OpenAI e Qdrant (creati solo se possibile)
# ─────────────────────────────────────────────

client_llm: Optional[OpenAI]
if OPENAI_API_KEY:
    client_llm = OpenAI(api_key=OPENAI_API_KEY)
else:
    client_llm = None

if QDRANT_URL and QDRANT_API_KEY:
    client: Optional[QdrantClient] = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10.0,
    )
else:
    client = None


# ─────────────────────────────────────────────
# Funzioni di utilità
# ─────────────────────────────────────────────


def _get_embedding(text: str) -> List[float]:
    """
    Calcola l'embedding della query usando OpenAI.

    Se client_llm non è configurato, solleva ValueError (gestito a monte).
    """
    if client_llm is None:
        raise ValueError(
            "[product_search] client_llm non configurato: manca OPENAI_API_KEY."
        )

    response = client_llm.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _build_filter(collection_filter: Optional[str]) -> Optional[Filter]:
    """
    Costruisce un filtro Qdrant opzionale sulla collection (campo 'collection').
    """
    if not collection_filter:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="collection",
                match=MatchValue(value=collection_filter),
            )
        ]
    )


def _normalize_product_point(
    payload: Dict[str, Any],
    score: float,
    user_query: str,
) -> Dict[str, Any]:
    """
    Normalizza il payload Qdrant in un dizionario prodotto coerente con l'API.

    Campi attesi nel payload:
    - id
    - name
    - url
    - description
    - image_url
    - sku
    - brand
    - price
    - currency
    - collection
    - features_text
    - tech_specs_text
    """
    product_id = payload.get("id") or payload.get("product_id")
    name = payload.get("name") or payload.get("title") or "Prodotto senza nome"
    url = payload.get("url") or payload.get("product_url")
    description = payload.get("description") or ""
    image_url = payload.get("image_url")
    sku = payload.get("sku")
    brand = payload.get("brand")
    price = payload.get("price")
    currency = payload.get("currency", "EUR")
    collection = payload.get("collection")
    features_text = payload.get("features_text")
    tech_specs_text = payload.get("tech_specs_text")

    if collection:
        reason = (
            f"Trovato il prodotto: {name} della collezione '{collection}' "
            f"del brand {brand or 'Scicon Sports'} con prezzo indicativo di {price} {currency} "
            f"in base alla tua richiesta: \"{user_query}\"."
        )
    else:
        reason = (
            f"Trovato il prodotto: {name} "
            f"del brand {brand or 'Scicon Sports'} con prezzo indicativo di {price} {currency} "
            f"in base alla tua richiesta: \"{user_query}\"."
        )

    return {
        "id": product_id,
        "name": name,
        "url": url,
        "description": description,
        "image_url": image_url,
        "sku": sku,
        "brand": brand or "Scicon Sports",
        "price": price,
        "currency": currency,
        "collection": collection,
        "features_text": features_text,
        "tech_specs_text": tech_specs_text,
        "score": float(score),
        "reason": reason,
    }


# ─────────────────────────────────────────────
# Funzione principale: search_products
# ─────────────────────────────────────────────


def search_products(
    query: str,
    top_k: int = 5,
    collection_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Esegue una ricerca semantica di prodotti su Qdrant a partire dalla query utente.

    Ritorna una lista di dizionari prodotto, ciascuno con:
    - id, name, url, description, image_url, sku, brand, price, currency, collection
    - features_text, tech_specs_text
    - score (float)
    - reason (string)
    """

    # Guard: se i client non sono configurati, non esplodere
    if client_llm is None:
        print(
            "[product_search] client_llm non configurato (manca OPENAI_API_KEY). "
            "Ritorno lista vuota."
        )
        return []

    if client is None:
        print(
            "[product_search] client Qdrant non configurato "
            "(manca QDRANT_URL o QDRANT_API_KEY). Ritorno lista vuota."
        )
        return []

    # Calcolo embedding della query
    try:
        embedding = _get_embedding(query)
    except Exception as e:
        print(f"[product_search] Errore nel calcolo embedding: {e!r}. Ritorno lista vuota.")
        return []

    # Costruisco il filtro opzionale
    qdrant_filter = _build_filter(collection_filter)

    # Eseguo la query su Qdrant con gestione errori
    try:
        search_result = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=embedding,
            query_filter=qdrant_filter,
            with_payload=True,
            limit=top_k,
        )
    except ResponseHandlingException as e:
        print(f"[product_search] Errore di risposta Qdrant: {e!r}. Ritorno lista vuota.")
        return []
    except Exception as e:
        print(f"[product_search] Errore generico Qdrant: {e!r}. Ritorno lista vuota.")
        return []

    points = getattr(search_result, "points", []) or []

    products: List[Dict[str, Any]] = []

    for p in points:
        payload = getattr(p, "payload", {}) or {}
        score = getattr(p, "score", 0.0) or 0.0

        normalized = _normalize_product_point(
            payload=payload,
            score=score,
            user_query=query,
        )
        products.append(normalized)

    return products
