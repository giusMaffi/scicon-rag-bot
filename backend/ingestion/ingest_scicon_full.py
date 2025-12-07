"""
ingest_scicon_full.py

Ingestion FULL per Scicon RAG Bot.

Obiettivo:
- Crawl dell'intero sito Scicon (entro un limite di pagine configurabile).
- Estrazione del testo principale dalle pagine HTML.
- Chunking del contenuto.
- Embedding via OpenAI.
- Upsert in Qdrant nella collection indicata da QDRANT_COLLECTION.

NOTE:
- Per ora è una ingestion "general purpose": carichiamo TUTTE le pagine
  (PDP + contenuti, blog, ecc.) all'interno della stessa collection.
- In uno step successivo possiamo raffinare tagging, linking e logica di retrieval.
"""

import os
import re
import uuid
import time
from datetime import datetime
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from openai import OpenAI
from dotenv import load_dotenv


# ==========================
# CONFIGURAZIONE BASE
# ==========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scicon_full")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY non impostata.")
if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL non impostata.")
if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY non impostata.")

# Dominio di base da cui partire (adatta se usi una lingua specifica)
BASE_URL = "https://sciconsports.com/"

# Limite massimo di pagine da craware in questo run (ridotto per test)
MAX_PAGES = 40

# Rate limit (secondi tra una richiesta e la successiva, per non massacrare il sito)
REQUEST_SLEEP = 0.3

# Percorsi da includere (puoi restringere se vuoi)
ALLOWED_PATH_PREFIXES = [
    "/",
]

# Percorsi da escludere (checkout, account, carrello, ecc.)
EXCLUDED_PATH_PATTERNS = [
    r"/cart",
    r"/checkout",
    r"/account",
    r"/customer",
    r"/login",
    r"/wishlist",
    r"/search",
    r"/admin",
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|pdf|zip|mp4|mov)$",
]


# ==========================
# UTILITY
# ==========================

client = OpenAI(api_key=OPENAI_API_KEY)


def is_allowed_url(url: str, base_domain: str) -> bool:
    """
    Verifica che l'URL:
    - appartenga allo stesso dominio
    - non matchi pattern esclusi
    - rientri nei path permessi
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.netloc and parsed.netloc != base_domain:
        return False

    path = parsed.path or "/"

    # Esclusioni
    for pattern in EXCLUDED_PATH_PATTERNS:
        if re.search(pattern, path):
            return False

    # Inclusioni
    if not any(path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return False

    return True


def fetch_html(url: str) -> str:
    """
    Effettua una GET e restituisce l'HTML in testo.
    """
    headers = {
        "User-Agent": "Scicon-RAG-Bot-Ingestion/1.0 (+https://sciconsports.com)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_main_text(html: str) -> Dict[str, str]:
    """
    Estrae titolo e testo principale dalla pagina HTML.
    Approccio semplice ma robusto; si può raffinare in futuro.
    """
    # 🔧 Parser cambiato da "lxml" a "html.parser"
    soup = BeautifulSoup(html, "html.parser")

    # Titolo
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Proviamo a prendere l'elemento <main>, se presente
    main = soup.find("main")
    if main:
        candidate = main
    else:
        # Fallback: body
        candidate = soup.body or soup

    # Rimuoviamo aree chiaramente non utili
    for tag in candidate.find_all(
        ["nav", "footer", "script", "style", "noscript", "header", "form"]
    ):
        tag.decompose()

    text = candidate.get_text(separator="\n", strip=True)
    # Cleanup di base
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return {"title": title, "text": text}


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    """
    Chunking semplice basato sui caratteri, con overlap.
    """
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0
        if start >= length:
            break

    return chunks


def get_embedding(text: str) -> List[float]:
    """
    Calcola embedding con OpenAI.
    Usa text-embedding-3-small (1536 dimensioni).
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ==========================
# QDRANT SETUP
# ==========================

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def ensure_collection(client_q: QdrantClient, vector_size: int = 1536):
    """
    Crea la collection se non esiste.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client_q.get_collection(QDRANT_COLLECTION)
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' già esistente.")
    except UnexpectedResponse:
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' non trovata. La creo...")
        client_q.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' creata.")


def upsert_chunks(
    client_q: QdrantClient,
    url: str,
    title: str,
    chunks: List[str],
):
    """
    Upsert di una lista di chunk per una singola pagina.
    """
    points: List[PointStruct] = []
    timestamp = datetime.utcnow().isoformat()

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        point_id = str(uuid.uuid4())

        payload = {
            "url": url,
            "title": title,
            "chunk_index": idx,
            "text": chunk,
            "doc_type": "web_page",
            "source": "scicon_full_crawl",
            "ingested_at": timestamp,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    if not points:
        return

    client_q.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print(f"[QDRANT] Inseriti {len(points)} chunk per {url}")


# ==========================
# CRAWLER
# ==========================

def crawl_site(start_url: str, max_pages: int) -> List[str]:
    """
    Crawler BFS semplice che ritorna la lista di URL HTML da processare.
    """
    parsed_base = urlparse(start_url)
    base_domain = parsed_base.netloc

    to_visit = [start_url]
    visited: Set[str] = set()
    result_urls: List[str] = []

    while to_visit and len(result_urls) < max_pages:
        current = to_visit.pop(0)

        if current in visited:
            continue
        visited.add(current)

        if not is_allowed_url(current, base_domain):
            continue

        print(f"[CRAWL] Fetch -> {current}")
        try:
            html = fetch_html(current)
        except Exception as e:
            print(f"[CRAWL] Errore nel fetch di {current}: {e}")
            continue

        result_urls.append(current)

        # Estraggo nuovi link
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href:
                    continue

                abs_url = urljoin(current, href)
                if abs_url not in visited and abs_url not in to_visit:
                    if is_allowed_url(abs_url, base_domain):
                        to_visit.append(abs_url)
        except Exception as e:
            print(f"[CRAWL] Errore nell'analisi link di {current}: {e}")

        time.sleep(REQUEST_SLEEP)

    print(f"[CRAWL] Trovate {len(result_urls)} pagine da processare.")
    return result_urls


def process_urls(urls: List[str]):
    """
    Estrae il contenuto, chunkizza, embedda e invia a Qdrant.
    """
    qdrant = get_qdrant_client()
    ensure_collection(qdrant)

    for i, url in enumerate(urls, start=1):
        print(f"[PAGE {i}/{len(urls)}] {url}")
        try:
            html = fetch_html(url)
            data = extract_main_text(html)
            title = data["title"]
            text = data["text"]

            if not text or len(text) < 200:
                print(f"[SKIP] Contenuto troppo breve per {url}")
                continue

            chunks = chunk_text(text, max_chars=1200, overlap=200)
            if not chunks:
                print(f"[SKIP] Nessun chunk generato per {url}")
                continue

            upsert_chunks(qdrant, url, title, chunks)

        except Exception as e:
            print(f"[ERROR] Errore durante la processazione di {url}: {e}")


def main():
    print("=== SCICON RAG BOT – INGESTION FULL ===")
    print(f"BASE_URL: {BASE_URL}")
    print(f"MAX_PAGES: {MAX_PAGES}")
    print(f"QDRANT_COLLECTION: {QDRANT_COLLECTION}")
    print("========================================")

    urls = crawl_site(BASE_URL, MAX_PAGES)
    process_urls(urls)
    print("=== Ingestion FULL completata. ===")


if __name__ == "__main__":
    main()
