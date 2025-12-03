# SCICON RAG BOT – MASTER DOCUMENT (v2, clean slate)

## 1. Visione del progetto

SCICON RAG BOT è il backend intelligente per:
- un **bot conversazionale** sul sito Scicon Sports,
- con capacità di **Product Finder** avanzato,
- basato su **RAG (Retrieval Augmented Generation)**,
- focalizzato inizialmente sui **prodotti** (PDP) del catalogo.

Obiettivo:
- aiutare l’utente a trovare i prodotti giusti (occhiali, borse, abbigliamento, ecc.)
- rispondendo in linguaggio naturale,
- usando dati reali del sito Scicon (descrizioni + tab tecnici).

---

## 2. Stack tecnologico

- **Linguaggio**: Python 3
- **Web framework**: FastAPI
- **RAG DB**: Qdrant (locale o cloud)
- **LLM provider**: OpenAI (GPT per risposta, embeddings per ricerca semantica)
- **Embeddings**: `text-embedding-3-large`
- **Test**: pytest + TestClient FastAPI
- **Deploy**: da definire (es. container Docker, integrazione con widget sul sito)

---

## 3. Struttura delle cartelle (v2 ufficiale)

```text
scicon-rag-bot/
├─ backend/
│  ├─ __init__.py
│  ├─ app.py                 # FastAPI app principale
│  └─ rag/
│     ├─ __init__.py
│     └─ product_search.py   # logica RAG per ricerca prodotti
├─ ingestion/
│  ├─ __init__.py
│  ├─ data/                  # file sorgente (es. link_prodotti_scicon.xlsx)
│  └─ ingest_scicon_products.py  # script ingestion catalogo prodotti
├─ config/
│  ├─ __init__.py
│  ├─ settings.py            # configurazione centralizzata (env)
│  └─ .env.example           # template variabili ambiente
├─ docs/
│  └─ MASTER_SCICON_RAG_BOT.md  # questo file
├─ tests/
│  ├─ __init__.py
│  └─ test_health.py         # test di base health check API
├─ scripts/
│  └─ run_api_dev.sh         # script avvio FastAPI in dev
├─ requirements.txt
├─ README.md
└─ .gitignore
