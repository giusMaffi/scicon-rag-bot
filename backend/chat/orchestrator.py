"""
backend.chat.orchestrator

Orchestratore centrale della chat del bot SCICON.

Per ora:
- instrada tutte le richieste al motore di raccomandazione prodotti
  (build_product_advice), che restituisce già il BOT CONTRACT.

In futuro:
- potrà decidere se usare RAG prodotti, RAG contenuti, FAQ, ecc.
"""

from typing import Any, Dict, Optional

from backend.chat.product_advisor import build_product_advice


def orchestrate_chat(
    user_query: str,
    top_k: int = 5,
    channel: Optional[str] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestratore principale.

    Parametri:
    - user_query: testo inserito dall'utente
    - top_k: quanti risultati massimi usare per la risposta
    - channel: canale di provenienza (es. "site_widget", "whatsapp", ecc.) [opzionale]
    - collection: eventuale filtro di collezione prodotti [opzionale]

    Oggi:
    - routing fisso verso il product advisor (raccomandazione prodotti)

    Domani:
    - logica di routing basata su intent (prodotti, contenuti, FAQ, ecc.)
    """

    # Per ora, routing fisso verso la raccomandazione prodotti.
    # Il product advisor restituisce già un dizionario conforme al BOT CONTRACT.
    advice = build_product_advice(
        user_query=user_query,
        top_k=top_k,
        collection_filter=collection,
    )

    # Possiamo arricchire il meta con info sul canale, se presente.
    meta = advice.get("meta", {}) or {}
    applied_filters = meta.get("applied_filters", {}) or {}

    # Aggiorniamo eventuali metadati senza rompere la struttura.
    meta["applied_filters"] = applied_filters
    if channel:
        # Aggiungiamo il canale come informazione di contesto nel meta.
        meta["channel"] = channel

    advice["meta"] = meta

    return advice
