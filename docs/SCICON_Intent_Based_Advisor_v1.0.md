📘 Documento di Riallineamento – Progetto SCICON Intent Advisor
Nome progetto (funzionale): Intent-Based Product Advisor for SCICON
Versione: v1.0
Stato: Attivo
Obiettivo: Arrivare a una soluzione reale, funzionante e osservabile per SCICON, basata sugli intenti e utilizzabile come product advisor con analytics su cosa gli utenti cercano davvero.

1. Contesto e decisioni chiave
1.1 Da “prodotto generico” a soluzione verticale

Fino ad oggi il lavoro fatto su Intento / SCICON è stato molto ampio (RAG, intent detection, assistant, ecc.), ma poco “chiuso” in una forma vendibile.

La decisione strategica è:

NON costruire subito una piattaforma generica o un “product finder universale”.

SÌ concentrarsi su SCICON come caso verticale reale, con:

una journey chiara,

un assistant basato sugli intenti,

analytics strutturati sugli intenti di acquisto.

1.2 Reset di direzione, NON reset di lavoro

Niente viene buttato.

Tutto il lavoro su intent detection, RAG, assistant conversazionale e logiche di recommendation viene considerato asset.

Il cambiamento è di focus:

→ da “prodotto generico”
→ a soluzione concreta per SCICON.

2. Scopo del progetto
2.1 Cosa stiamo costruendo

Una soluzione conversazionale che:

riconosce l’intento di acquisto dell’utente,

lo guida con domande progressive,

riduce l’ansia decisionale,

arriva a 1–2 raccomandazioni motivate,

registra come e cosa gli utenti cercano.

Nome funzionale: Intent-Based Product Advisor for SCICON

2.2 Cosa NON stiamo costruendo (per ora)

Fuori scope:

chatbot generico multi-uso,

piattaforma AI multi-settore,

SaaS multi-tenant,

UI perfetta,

integrazioni enterprise complete.

3. Focus iniziale: una sola journey SCICON
3.1 Categoria e scenario scelto

Brand: SCICON

Categoria: occhiali da ciclismo

Scenario: uscite lunghe con luce variabile / diffusa

Query tipo:
“Cerco occhiali da ciclismo per uscite lunghe, spesso con luce variabile o cielo coperto, vorrei qualcosa di versatile.”

Motivi della scelta:

realistica e frequente,

complessa quanto basta,

perfetta per mostrare valore dell’AI.

4. Logica dell’Intent-Based Product Advisor
4.1 Intenti rilevanti

Valutazione / esplorazione

Comparazione

Riduzione del rischio

Affidabilità tecnica

L’intento è una variabile di stato, non solo una label.

4.2 Flusso conversazionale ideale

Input utente

Intent detection

Risposta di apertura

Domande progressive

Q1 terreno

Q2 variazione luce

Q3 priorità

Esclusione opzioni (esplicitata)

Raccomandazione finale

1 prodotto principale

1 alternativa

Call to action

5. Analytics sugli intenti
5.1 Unità base: sessione

Campi minimi: session_id, timestamp, lingua, canale.

5.2 Eventi da loggare

session_start

query_iniziale

intent_detected

question_asked

answer_given

option_excluded

recommendation_shown

session_end

Formato:

{
  "timestamp": "...",
  "session_id": "...",
  "event_type": "event_name",
  "data": { ... }
}

5.3 Cosa diventa possibile

capire pattern reali di ricerca,

mappare gli intenti più frequenti,

capire cosa genera esclusione,

identificare contenuti deboli,

migliorare prodotti e journey.

Il sistema diventa un osservatorio sugli intenti di acquisto.

6. Ruoli nel progetto
6.1 Ruolo di Giuseppe

Product / Intent Architect

Prompt / Logic Designer

6.2 Ruolo “sviluppatore” (coperto da AI assistant)

progettazione tecnica,

endpoint,

logging,

codice completo,

istruzioni passo passo.

7. Next steps (alto livello)

Salvare questo documento come v1.0.

Progettare l’MVP tecnico:

endpoint /advisor/scicon,

flusso base query → intent → domande → output,

logging JSONL o SQLite.

Sviluppare il prototipo locale.

Test qualitativi.

Iniziare a raccogliere insight reali.

Nota finale

Questo documento non azzera il lavoro passato, ma lo ricompone in una direzione chiara:

➡️ SCICON come primo caso reale di Intent-Based Product Advisor con analytics sugli intenti di acquisto.