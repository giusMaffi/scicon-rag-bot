# backend/rag/product_search.py

"""
Motore di ricerca prodotti per il SCICON RAG BOT (S3).

Logica:
- Usa Qdrant come vettore store per i prodotti.
- Usa OpenAI Embeddings per trasformare la query utente in vettore.
- Esegue una query semantica (query_points) con un top_k abbastanza ampio.
- Applica un re-ranking "intelligente" basato su:
  - tipo di query (gravel, strada, mtb, borse porta bici, casual, ecc.),
  - metadata del prodotto (collection, url, name, ecc.),
  - penalizzazione di prodotti lifestyle/outlet quando l'intento è performance,
  - boost dei modelli performance quando il contesto è tecnico.

S3 – Miglioria chiave:
- Se l'utente chiede qualcosa di chiaramente "gravel + performance"
  e nei top_k restituiti compaiono solo occhiali lifestyle (es. GRAVEL)
  ma nessun occhiale performance, viene eseguito un SECONDO PASSAGGIO
  usando una query di dominio più esplicita (occhiali da ciclismo performance).
- Se nel secondo pass compaiono occhiali performance nei top_k, questi
  risultati vengono preferiti al primo pass.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from pydantic import BaseModel

# --------------------------------------------------------------------
# Caricamento .env e inizializzazione client
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

print(f"[product_search] Uso .env da: {ENV_PATH}")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scicon_products")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not QDRANT_URL:
    raise RuntimeError("[product_search] QDRANT_URL non impostata in .env")

if not OPENAI_API_KEY:
    raise RuntimeError("[product_search] OPENAI_API_KEY non impostata in .env")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------------------------
# Modello Product (usato per type hints e/o response_model)
# --------------------------------------------------------------------

class Product(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
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


# --------------------------------------------------------------------
# Helpers: normalizzazione e flag query
# --------------------------------------------------------------------

def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).strip()


def _detect_query_flags(query: str) -> Dict[str, bool]:
    """
    Rileva intenzioni macro dalla query utente.
    Non è un intent classifier LLM, ma un set di euristiche di parole chiave.
    Servirà per il re-ranking e per decidere il secondo passaggio.
    """
    q = query.lower()

    is_gravel = "gravel" in q
    is_mtb = any(kw in q for kw in ["mtb", "mountain bike", "trail", "enduro"])
    is_road = any(
        kw in q
        for kw in ["bici da strada", "strada", "road bike", "corsa su strada"]
    )
    is_casual = any(
        kw in q
        for kw in ["casual", "vita quotidiana", "everyday", "lifestyle", "città"]
    )
    is_travel_bag = any(
        kw in q
        for kw in [
            "borsa porta bici",
            "bike bag",
            "valigia porta bici",
            "viaggi in aereo",
            "bike travel",
        ]
    )
    is_performance = any(
        kw in q
        for kw in [
            "performance",
            "gara",
            "race",
            "allenamenti lunghi",
            "uscite lunghe",
            "competizione",
        ]
    )

    # Se parla di gravel/mtb/road con uscita lunga, la trattiamo come performance anche se non c'è la parola
    if (is_gravel or is_mtb or is_road) and any(
        kw in q for kw in ["uscite lunghe", "allenamenti", "allenamento", "gran fondo", "lunghe ore"]
    ):
        is_performance = True

    return {
        "is_gravel": is_gravel,
        "is_mtb": is_mtb,
        "is_road": is_road,
        "is_casual": is_casual,
        "is_travel_bag": is_travel_bag,
        "is_performance": is_performance,
    }


# --------------------------------------------------------------------
# Helpers: classificazione ruolo prodotto
# --------------------------------------------------------------------

def _classify_product_role(payload: Dict[str, Any]) -> str:
    """
    Classifica il "ruolo" del prodotto (eyewear performance, eyewear lifestyle, bike bag, altro)
    usando campi che sappiamo essere presenti nel payload (collection, url, name, ecc.).
    """
    collection = _normalize_text(payload.get("collection")).lower()
    url = _normalize_text(payload.get("url")).lower()
    name = _normalize_text(payload.get("name")).lower()
    sku = _normalize_text(payload.get("sku")).lower()

    # Borse / valigie porta bici
    if any(
        kw in collection or kw in url or kw in name
        for kw in ["bike-bag", "bike-bags", "valigia", "porta-bici", "bike-travel"]
    ):
        return "bike_bag"

    # Occhiali GRAVEL / Vertec / outlet lifestyle
    if "gravel" in name or sku.startswith("ey270") or "/gravel-" in url:
        return "eyewear_lifestyle"
    if "vertec" in url or "outlet-occhiali" in collection:
        return "eyewear_lifestyle"

    # Occhiali performance (Aeroshade, Aerowing, Aerotech, Kunken, ecc.)
    performance_keywords = [
        "aeroshade",
        "aerowing",
        "aerotech",
        "kunken",
        "aerojet",
        "occhiali-da-ciclismo",
    ]
    if any(kw in name or kw in url or kw in collection for kw in performance_keywords):
        return "eyewear_performance"

    # Altri occhiali: se collection è "occhiali-da-ciclismo", assumiamo performance
    if "occhiali-da-ciclismo" in collection:
        return "eyewear_performance"

    return "other"


def _adjust_score_for_query(
    base_score: float,
    payload: Dict[str, Any],
    query_flags: Dict[str, bool],
) -> float:
    """
    Modifica lo score Qdrant in base a:
    - tipo di query (gravel/mtb/road/performance/casual/travel_bag),
    - ruolo del prodotto (performance / lifestyle / bag / altro).
    """
    role = _classify_product_role(payload)
    score = base_score

    is_gravel = query_flags["is_gravel"]
    is_mtb = query_flags["is_mtb"]
    is_road = query_flags["is_road"]
    is_casual = query_flags["is_casual"]
    is_travel_bag = query_flags["is_travel_bag"]
    is_performance = query_flags["is_performance"]

    # 1) Query gravel/mtb/road performance → vogliamo occhiali performance, NON lifestyle
    if (is_gravel or is_mtb or is_road) and is_performance and not is_casual:
        if role == "eyewear_performance":
            score += 0.10  # boost modelli performance
        if role == "eyewear_lifestyle":
            score -= 0.15  # penalizza GRAVEL / outlet / lifestyle

    # 2) Query travel bag → vogliamo borse/valigie porta bici
    if is_travel_bag:
        if role == "bike_bag":
            score += 0.20  # forte preferenza
        else:
            score -= 0.10

    # 3) Query casual → GRAVEL/Vertec possono andare bene, performance leggermente penalizzati
    if is_casual and not is_performance and not is_travel_bag:
        if role == "eyewear_lifestyle":
            score += 0.10
        if role == "eyewear_performance":
            score -= 0.05

    return score


# --------------------------------------------------------------------
# OpenAI Embeddings
# --------------------------------------------------------------------

def _embed_query(query: str) -> List[float]:
    """
    Usa OpenAI Embeddings (API >= 1.0.0) per generare il vettore della query.
    """
    response = openai_client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        input=query,
    )
    return response.data[0].embedding


# --------------------------------------------------------------------
# Helper: singolo passaggio semantico su Qdrant
# --------------------------------------------------------------------

def _semantic_qdrant_search(
    base_query: str,
    user_query: str,
    top_k: int,
    collection_filter: Optional[str],
    query_flags: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Esegue un singolo passaggio di ricerca semantica su Qdrant,
    applica il re-ranking e ritorna:
    - bot_message
    - products (lista di dict)
    - follow_up_suggestions
    - meta
    - debug (info per decisione secondo pass)
    """
    # 1) Embedding della query di ricerca (che può essere raffinata rispetto a user_query)
    query_vector = _embed_query(base_query)

    # 2) Filtro opzionale per collezione
    qdrant_filter: Optional[Filter] = None
    if collection_filter:
        # ATTENZIONE: questo richiede un indice "keyword" su 'collection' in Qdrant
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="collection",
                    match=MatchValue(value=collection_filter),
                )
            ]
        )

    # 3) Parametri di ricerca
    search_params = SearchParams(
        hnsw_ef=128,
        exact=False,
    )

    limit_env = os.getenv("PRODUCTS_RAG_LIMIT")
    if limit_env:
        limit = int(limit_env)
    else:
        # usiamo almeno top_k ma con un minimo un po' più alto per dare spazio al re-ranking
        limit = max(top_k, 10)

    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,             # <-- API nuova: 'query', non 'query_vector'
        query_filter=qdrant_filter,
        limit=limit,
        search_params=search_params,
        with_payload=True,
        with_vectors=False,
    )

    points = results.points or []

    if not points:
        return {
            "bot_message": (
                "Al momento non trovo prodotti adatti a partire da questa descrizione. "
                "Se vuoi, prova a dirmi per che disciplina ti servono (strada, gravel, mtb) "
                "e in che condizioni di luce li userai più spesso."
            ),
            "products": [],
            "follow_up_suggestions": [
                "Ti servono occhiali per strada, gravel o mtb?",
                "Preferisci dare priorità al comfort o alla massima protezione della lente?",
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
            "debug": {
                "query_used": base_query,
                "has_performance_all": False,
                "has_lifestyle_all": False,
                "has_performance_topk": False,
                "has_lifestyle_topk": False,
                "max_score": 0.0,
            },
        }

    # 4) Re-ranking con euristiche di dominio
    reranked: List[Dict[str, Any]] = []
    max_score = 0.0

    has_performance_all = False
    has_lifestyle_all = False

    for pt in points:
        payload = pt.payload or {}
        base_score = float(pt.score or 0.0)
        adjusted_score = _adjust_score_for_query(base_score, payload, query_flags)

        role = _classify_product_role(payload)
        if role == "eyewear_performance":
            has_performance_all = True
        if role == "eyewear_lifestyle":
            has_lifestyle_all = True

        if adjusted_score > max_score:
            max_score = adjusted_score

        reranked.append(
            {
                "score": base_score,
                "adjusted_score": adjusted_score,
                "payload": payload,
                "role": role,
            }
        )

    # 5) Ordina per adjusted_score decrescente e prendi i top_k
    reranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
    top_items = reranked[:top_k]

    products: List[Dict[str, Any]] = []

    has_performance_topk = False
    has_lifestyle_topk = False

    for item in top_items:
        payload = item["payload"]
        role = item["role"]

        if role == "eyewear_performance":
            has_performance_topk = True
        if role == "eyewear_lifestyle":
            has_lifestyle_topk = True

        image_val = payload.get("image_url")
        # Se image_url è un dict (JSON-LD), proviamo a estrarre la URL
        if isinstance(image_val, dict):
            image_url = image_val.get("url") or image_val.get("image")
        else:
            image_url = image_val

        products.append(
            {
                "id": payload.get("id"),
                "name": payload.get("name"),
                "url": payload.get("url"),
                "description": payload.get("description"),
                "image_url": image_url,
                "sku": payload.get("sku"),
                "brand": payload.get("brand"),
                "price": payload.get("price"),
                "currency": payload.get("currency"),
                "collection": payload.get("collection"),
                "features_text": payload.get("features_text"),
                "tech_specs_text": payload.get("tech_specs_text"),
                "score": float(item["adjusted_score"]),
                "reason": None,
            }
        )

    # 6) Messaggio generico e follow-up in base alla categoria
    bot_message = (
        "Ecco alcuni prodotti Scicon che potrebbero fare al caso tuo in base a quello che mi hai descritto."
    )

    follow_up_suggestions: List[str] = []

    if query_flags["is_travel_bag"]:
        follow_up_suggestions = [
            "Vuoi dare priorità alla protezione massima o alla leggerezza della borsa porta bici?",
            "Ti serve soprattutto per viaggi in aereo o anche in auto/treno?",
        ]
    elif query_flags["is_gravel"] or query_flags["is_mtb"] or query_flags["is_road"]:
        follow_up_suggestions = [
            "Preferisci una lente fotocromatica o più specifica per sole forte?",
            "Ti interessa anche avere un modello utilizzabile in contesto casual/quotidiano?",
            "Vuoi che ti suggerisca una soluzione più economica o il top di gamma?",
        ]
    else:
        follow_up_suggestions = [
            "Vuoi che ti suggerisca anche modelli più adatti al pieno sole?",
            "Preferisci dare priorità al comfort o alla massima protezione della lente?",
            "Ti interessa confrontare questi modelli anche in base al prezzo?",
        ]

    return {
        "bot_message": bot_message,
        "products": products,
        "follow_up_suggestions": follow_up_suggestions,
        "meta": {
            "intent": "product_recommendation",
            "user_query": user_query,
            "sources": ["products_rag"],
            "confidence_score": float(max_score),
            "applied_filters": {
                "collection": collection_filter,
                "price_range": None,
                "lens_type": None,
            },
        },
        "debug": {
            "query_used": base_query,
            "has_performance_all": has_performance_all,
            "has_lifestyle_all": has_lifestyle_all,
            "has_performance_topk": has_performance_topk,
            "has_lifestyle_topk": has_lifestyle_topk,
            "max_score": float(max_score),
        },
    }


# --------------------------------------------------------------------
# Funzione principale: search_products (con logica a due passaggi S3)
# --------------------------------------------------------------------

def search_products(
    query: str,
    top_k: int = 5,
    collection_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Funzione principale usata dall'endpoint /chat/products e da product_advisor.

    Parametri:
    - query: testo naturale dell'utente.
    - top_k: numero massimo di prodotti da restituire.
    - collection_filter: eventuale filtro di collezione (al momento spesso None).

    Ritorna:
    {
      "bot_message": "...",
      "products": [ {...}, ... ],
      "follow_up_suggestions": [...],
      "meta": {...}
    }

    S3 – Comportamento speciale gravel+performance:
    - Se l'utente chiede qualcosa di chiaramente gravel+performance
      e nei top_k del primo pass abbiamo solo lifestyle (es. GRAVEL)
      ma nessun occhiale performance, viene eseguito un SECONDO PASS.
    """
    user_query = query.strip()
    if not user_query:
        return {
            "bot_message": "Per aiutarti a trovare il prodotto giusto ho bisogno di qualche dettaglio in più.",
            "products": [],
            "follow_up_suggestions": [
                "Per che disciplina ti servono gli occhiali (strada, gravel, mtb)?",
                "In che condizioni di luce li userai più spesso?",
            ],
            "meta": {
                "intent": "product_recommendation",
                "user_query": user_query,
                "sources": [],
                "confidence_score": 0.0,
                "applied_filters": {
                    "collection": collection_filter,
                    "price_range": None,
                    "lens_type": None,
                },
            },
        }

    # 1) Flags di query (euristiche dominio)
    query_flags = _detect_query_flags(user_query)

    # 2) Primo pass: uso direttamente la query utente
    first_pass = _semantic_qdrant_search(
        base_query=user_query,
        user_query=user_query,
        top_k=top_k,
        collection_filter=collection_filter,
        query_flags=query_flags,
    )

    debug_1 = first_pass.get("debug", {})
    has_perf_topk_1 = debug_1.get("has_performance_topk", False)
    has_life_topk_1 = debug_1.get("has_lifestyle_topk", False)

    # 3) Decidere se serve un SECONDO PASS (gravel + performance ma solo lifestyle nei top_k)
    need_second_pass = (
        query_flags["is_gravel"]
        and query_flags["is_performance"]
        and has_life_topk_1
        and not has_perf_topk_1
    )

    if not need_second_pass:
        # restituiamo il primo pass senza il blocco debug
        return {
            "bot_message": first_pass["bot_message"],
            "products": first_pass["products"],
            "follow_up_suggestions": first_pass["follow_up_suggestions"],
            "meta": first_pass["meta"],
        }

    # 4) Secondo pass: query di dominio più esplicita per occhiali performance
    refined_query = (
        "occhiali da ciclismo performance per gravel e bici da strada, "
        "uscite lunghe, luce diffusa e cielo coperto, molto protettivi e versatili, "
        "con lenti fotocromatiche o adatte a luce variabile, non occhiali lifestyle."
    )

    second_pass = _semantic_qdrant_search(
        base_query=refined_query,
        user_query=user_query,  # manteniamo la query originale per il meta
        top_k=top_k,
        collection_filter=collection_filter,
        query_flags=query_flags,
    )

    debug_2 = second_pass.get("debug", {})
    has_perf_topk_2 = debug_2.get("has_performance_topk", False)

    # 5) Se nel secondo pass abbiamo almeno un prodotto performance nei top_k,
    #    preferiamo il secondo pass. Altrimenti restiamo sul primo.
    if has_perf_topk_2:
        chosen = second_pass
    else:
        chosen = first_pass

    return {
        "bot_message": chosen["bot_message"],
        "products": chosen["products"],
        "follow_up_suggestions": chosen["follow_up_suggestions"],
        "meta": chosen["meta"],
    }
