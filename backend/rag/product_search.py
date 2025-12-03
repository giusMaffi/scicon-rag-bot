"""
backend.rag.product_search

Logica di:
- caricamento configurazioni (.env a livello di progetto)
- connessione a Qdrant
- embedding della query utente
- ricerca semantica prodotti Scicon
- normalizzazione dei campi payload (in particolare image_url)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchValue


# =========================
# Caricamento .env progetto
# =========================

ROOT_DIR = Path(__file__).resolve().parents[2]
env_path = ROOT_DIR / ".env"

if env_path.exists():
    print(f"[product_search] Uso .env da: {env_path}")
    load_dotenv(env_path)
else:
    print("[product_search] Nessun .env trovato nella root del progetto")


# =========================
# Config da environment
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_PRODUCTS", "scicon_products")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    print("OPENAI_API_KEY non impostata: la generazione degli embeddings potrebbe fallire.")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("QDRANT_URL o QDRANT_API_KEY non impostati: la ricerca prodotti potrebbe fallire.")


# =========================
# Client esterni
# =========================

client_qdrant: Optional[QdrantClient] = None
if QDRANT_URL and QDRANT_API_KEY:
    client_qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10.0,
    )

client_openai: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client_openai = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Modello di dominio Product
# =========================

@dataclass
class Product:
    id: str
    name: str
    url: str
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


# =========================
# Funzioni di supporto
# =========================

def _normalize_image_url(raw: Any) -> Optional[str]:
    """
    Normalizza il campo image_url proveniente dal payload Qdrant.

    Possibili formati:
    - stringa semplice: "https://..."
    - dict stile ImageObject:
        {
          "@type": "ImageObject",
          "url": "...",
          "image": "...",
          ...
        }
    - lista di stringhe o dict: [ ..., ... ]

    Restituisce sempre:
    - una stringa URL pulita, oppure
    - None se non si riesce a estrarre nulla di sensato.
    """
    if raw is None:
        return None

    # Caso semplice: già una stringa
    if isinstance(raw, str):
        val = raw.strip()
        return val or None

    # Caso lista: prendo il primo valore valido
    if isinstance(raw, list):
        for item in raw:
            candidate = _normalize_image_url(item)
            if candidate:
                return candidate
        return None

    # Caso dict / ImageObject
    if isinstance(raw, dict):
        # Provo campi tipici in ordine di preferenza
        for key in ("url", "image", "src"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    # Tipo imprevisto
    return None


def _embed_query(query: str) -> List[float]:
    """
    Calcola l'embedding della query utente usando il modello configurato.
    """
    if client_openai is None:
        raise RuntimeError(
            "Client OpenAI non configurato: impossibile calcolare gli embeddings per la query."
        )

    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def _build_filter(collection_filter: Optional[str]) -> Optional[Filter]:
    """
    Costruisce un filtro Qdrant opzionale sulla base della collezione Shopify
    (es: 'occhiali-da-ciclismo', 'aerotrail-sunglasses', ecc.)
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


# =========================
# Funzione principale
# =========================

def search_products(
    query: str,
    top_k: int = 5,
    collection_filter: Optional[str] = None,
) -> List[Product]:
    """
    Esegue una ricerca semantica dei prodotti Scicon in Qdrant
    a partire dalla query utente.

    - Calcola l'embedding della query
    - Chiama Qdrant .query_points(...)
    - Mappa il payload in oggetti Product
    - Normalizza i campi più “sporchi” (image_url in primis)
    """

    if client_qdrant is None:
        print("[product_search] ❌ Qdrant client non configurato.")
        return []

    try:
        query_vector = _embed_query(query)
    except Exception as e:
        print(f"[product_search] ❌ Errore nel calcolo embeddings: {e}")
        return []

    qdrant_filter = _build_filter(collection_filter)

    try:
        search_result = client_qdrant.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    except ResponseHandlingException as e:
        print("[product_search] ❌ Errore nella query a Qdrant:", e)
        if hasattr(e, "response") and getattr(e.response, "content", None):
            print("Raw response content:")
            print(e.response.content)
        return []
    except Exception as e:
        print("[product_search] ❌ Errore generico nella query a Qdrant:", e)
        return []

    products: List[Product] = []

    for point in search_result.points:
        payload: Dict[str, Any] = point.payload or {}

        # Campi semplici
        pid = payload.get("id") or str(point.id)
        name = payload.get("name") or payload.get("product_name") or ""
        url = payload.get("url") or ""
        description = payload.get("description") or payload.get("body_html") or None
        sku = payload.get("sku")
        brand = payload.get("brand")
        price = None
        currency = None
        collection = payload.get("collection")
        features_text = payload.get("features_text")
        tech_specs_text = payload.get("tech_specs_text")
        reason = payload.get("reason")  # opzionale, se salvato in payload

        # Prezzo e valuta (se presenti)
        raw_price = payload.get("price")
        if isinstance(raw_price, (int, float)):
            price = float(raw_price)
        elif isinstance(raw_price, str):
            try:
                price = float(raw_price.replace(",", "."))
            except ValueError:
                price = None

        raw_currency = payload.get("currency")
        if isinstance(raw_currency, str):
            currency = raw_currency

        # ✅ Normalizzazione robusta di image_url
        raw_image = payload.get("image_url") or payload.get("image")
        image_url = _normalize_image_url(raw_image)

        product = Product(
            id=pid,
            name=name,
            url=url,
            description=description,
            image_url=image_url,
            sku=sku,
            brand=brand,
            price=price,
            currency=currency,
            collection=collection,
            features_text=features_text,
            tech_specs_text=tech_specs_text,
            score=point.score,
            reason=reason,
        )
        products.append(product)

    return products
