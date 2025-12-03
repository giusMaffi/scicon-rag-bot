"""
ingestion.ingest_scicon_products

Ingestion prodotti Scicon a partire da una lista di URL di PDP (Excel).

Pipeline (versione from-scratch, no riciclo):
- Legge il file Excel con i link.
- Scarica l'HTML di ogni PDP.
- Estrae dati prodotto da JSON-LD (@type: Product).
- Estrae contenuti aggiuntivi dai TAB (specifiche tecniche, features, ecc.).
- Costruisce un payload prodotto completo.
- Genera embedding con OpenAI.
- Inserisce tutto in Qdrant nella collezione 'scicon_products'.

Prerequisiti:
- Variabili d'ambiente:
    OPENAI_API_KEY
    QDRANT_URL
    QDRANT_API_KEY (se necessario)
    QDRANT_PRODUCTS_COLLECTION (opzionale, default: scicon_products)
    EMBEDDING_MODEL (opzionale, default: text-embedding-3-large)

- Librerie:
    pip install requests beautifulsoup4 pandas openpyxl qdrant-client openai python-dotenv
"""

import os
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv  # <-- per leggere il file .env


# ---------- CARICAMENTO .ENV ----------

# Root del progetto (cartella scicon-rag-bot)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)


# ---------- CONFIGURAZIONE BASE ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LINKS_XLSX_PATH = os.path.join(
    BASE_DIR,
    "data",
    "link_prodotti_scicon.xlsx",
)

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_PRODUCTS_COLLECTION", "scicon_products")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

REQUEST_TIMEOUT = 15  # secondi
REQUEST_SLEEP = 0.5   # pausa tra richieste per non stressare il sito

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


# ---------- UTILITY: CARICAMENTO URL ----------

def load_urls_from_excel(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File Excel non trovato: {path}")

    df = pd.read_excel(path)
    urls = df.iloc[:, 0].dropna().astype(str).tolist()
    # Rimuove duplicati preservando l'ordine
    urls = list(dict.fromkeys(urls))
    logging.info(f"Caricati {len(urls)} URL unici dal file Excel.")
    return urls


# ---------- DOWNLOAD HTML ----------

def fetch_html(url: str) -> Optional[str]:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        if resp.status_code != 200:
            logging.warning(f"URL {url} -> status {resp.status_code}")
            return None
        return resp.text
    except Exception as e:
        logging.error(f"Errore nel download di {url}: {e}")
        return None


# ---------- PARSING JSON-LD PRODUCT ----------

def extract_product_from_ld_json(html: str, url: str) -> Optional[Dict]:
    """
    Estrae i dati prodotto da JSON-LD (@type: Product).
    Funziona bene per siti Shopify.
    """
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    product_obj = None

    for script in scripts:
        raw = script.string or script.text
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            if item_type == "Product" or (isinstance(item_type, list) and "Product" in item_type):
                product_obj = item
                break
        if product_obj:
            break

    if not product_obj:
        logging.warning(f"Nessun JSON-LD Product trovato per {url}")
        return None

    name = product_obj.get("name")
    description = product_obj.get("description")

    images = product_obj.get("image")
    if isinstance(images, list):
        image_url = images[0]
    else:
        image_url = images

    sku = product_obj.get("sku")
    brand = product_obj.get("brand")
    if isinstance(brand, dict):
        brand_name = brand.get("name")
    else:
        brand_name = brand

    offers = product_obj.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price = offers.get("price")
    currency = offers.get("priceCurrency")

    collection = None
    if "/collections/" in url:
        try:
            after = url.split("/collections/")[1]
            collection = after.split("/products/")[0]
        except Exception:
            collection = None

    slug = url.rstrip("/").split("/")[-1]

    product = {
        "id": slug,  # chiave umana
        "url": url,
        "name": name,
        "description": description,
        "image_url": image_url,
        "sku": sku,
        "brand": brand_name,
        "price": price,
        "currency": currency,
        "collection": collection,
        "slug": slug,
        "source": "sciconsports.com",
    }

    return product


# ---------- PARSING TAB TECNICI & SEZIONI AGGIUNTIVE ----------

HEADING_KEYWORDS_IT = [
    "specifiche", "specifiche tecniche", "caratteristiche", "dettagli",
    "dettagli prodotto", "tecnologia", "materiali", "cura", "istruzioni",
    "guida alle taglie", "taglie", "vestibilità",
]

HEADING_KEYWORDS_EN = [
    "specifications", "technical specifications", "features", "details",
    "product details", "technology", "materials", "care", "instructions",
    "size guide", "sizes", "fit",
]


def is_heading_tag(tag) -> bool:
    return tag.name in ["h2", "h3", "h4"]


def heading_matches_keywords(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    for kw in HEADING_KEYWORDS_IT + HEADING_KEYWORDS_EN:
        if kw in t:
            return True
    return False


def extract_section_text_from_heading(heading) -> str:
    """
    Dato un tag heading (h2/h3/h4), prova a estrarre il testo
    della sezione corrispondente (paragrafi, liste, tabelle).
    """
    parts: List[str] = []

    def extract_table_text(table_tag) -> List[str]:
        rows_text = []
        for row in table_tag.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if not cells:
                continue
            if len(cells) == 1:
                rows_text.append(cells[0])
            else:
                rows_text.append(f"{cells[0]}: {' '.join(cells[1:])}")
        return rows_text

    sib = heading.find_next_sibling()
    siblings_checked = 0
    while sib and siblings_checked < 6:
        siblings_checked += 1

        if is_heading_tag(sib):
            break

        for p in sib.find_all("p", recursive=True):
            txt = p.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

        for li in sib.find_all("li", recursive=True):
            txt = li.get_text(" ", strip=True)
            if txt:
                parts.append(f"- {txt}")

        for table in sib.find_all("table", recursive=True):
            rows = extract_table_text(table)
            parts.extend(rows)

        sib = sib.find_next_sibling()

    if not parts:
        parent = heading.parent
        if parent:
            for p in parent.find_all("p", recursive=True):
                txt = p.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)
            for li in parent.find_all("li", recursive=True):
                txt = li.get_text(" ", strip=True)
                if txt:
                    parts.append(f"- {txt}")
            for table in parent.find_all("table", recursive=True):
                rows = extract_table_text(table)
                parts.extend(rows)

    seen = set()
    unique_parts = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return "\n".join(unique_parts)


def extract_additional_sections(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    features_blocks: List[str] = []
    specs_blocks: List[str] = []

    headings = soup.find_all(is_heading_tag)

    for h in headings:
        text = (h.get_text(" ", strip=True) or "").lower()
        if not heading_matches_keywords(text):
            continue

        section_text = extract_section_text_from_heading(h)
        if not section_text:
            continue

        if "spec" in text or "specifiche" in text or "technical" in text:
            specs_blocks.append(section_text)
        else:
            features_blocks.append(section_text)

    features_text = "\n\n".join(features_blocks).strip()
    tech_specs_text = "\n\n".join(specs_blocks).strip()

    return features_text, tech_specs_text


# ---------- EMBEDDING ----------

def build_embedding_text(product: Dict) -> str:
    parts: List[str] = []

    if product.get("name"):
        parts.append(product["name"])

    if product.get("description"):
        parts.append(product["description"])

    if product.get("features_text"):
        parts.append("Caratteristiche principali:\n" + product["features_text"])

    if product.get("tech_specs_text"):
        parts.append("Specifiche tecniche:\n" + product["tech_specs_text"])

    meta_parts: List[str] = []
    if product.get("collection"):
        meta_parts.append(f"Collection: {product['collection']}")
    if product.get("brand"):
        meta_parts.append(f"Brand: {product['brand']}")
    if product.get("sku"):
        meta_parts.append(f"SKU: {product['sku']}")
    if product.get("price") and product.get("currency"):
        meta_parts.append(f"Price: {product['price']} {product['currency']}")

    if meta_parts:
        parts.append(" | ".join(meta_parts))

    return "\n\n".join(parts)


def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


# ---------- QDRANT ----------

def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_qdrant_collection(client: QdrantClient):
    logging.info(f"Ricreo la collezione Qdrant '{QDRANT_COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072, # dimensione corretta per text-embedding-3-large
            distance=Distance.COSINE,
        ),
    )


def upsert_products_to_qdrant(
    client: QdrantClient,
    products: List[Dict],
    embeddings: List[List[float]],
):
    assert len(products) == len(embeddings)
    points: List[PointStruct] = []

    for product, vector in zip(products, embeddings):
        point_id = uuid.uuid4().hex
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=product,
            )
        )

    logging.info(f"Upsert di {len(points)} prodotti in Qdrant...")
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
    )
    logging.info("Upsert completato.")


# ---------- MAIN PIPELINE ----------

def main():
    openai_client = OpenAI()
    qdrant_client = get_qdrant_client()
    ensure_qdrant_collection(qdrant_client)

    urls = load_urls_from_excel(LINKS_XLSX_PATH)
    logging.info("Inizio ingestion prodotti Scicon (from scratch, con tab tecnici)...")

    products_batch: List[Dict] = []
    embeddings_batch: List[List[float]] = []
    BATCH_SIZE = 40

    for idx, url in enumerate(urls, start=1):
        logging.info(f"[{idx}/{len(urls)}] Elaboro {url}")

        html = fetch_html(url)
        if not html:
            continue

        product = extract_product_from_ld_json(html, url)
        if not product:
            continue

        features_text, tech_specs_text = extract_additional_sections(html)
        product["features_text"] = features_text or None
        product["tech_specs_text"] = tech_specs_text or None

        text_for_embedding = build_embedding_text(product)

        try:
            vector = embed_text(openai_client, text_for_embedding)
        except Exception as e:
            logging.error(f"Errore embedding per {url}: {e}")
            continue

        products_batch.append(product)
        embeddings_batch.append(vector)

        if len(products_batch) >= BATCH_SIZE:
            upsert_products_to_qdrant(qdrant_client, products_batch, embeddings_batch)
            products_batch = []
            embeddings_batch = []

        time.sleep(REQUEST_SLEEP)

    if products_batch:
        upsert_products_to_qdrant(qdrant_client, products_batch, embeddings_batch)

    logging.info("Ingestion prodotti Scicon completata.")


if __name__ == "__main__":
    main()
