# RAG asistent cestovnej kancelárie napojený na ontológiu

Lokálny RAG systém (Retrieval-Augmented Generation), ktorý odpovedá na otázky o zájazdoch,
právach cestujúcich a cestovnom poistení. Kombinuje:

- **~10 doménových PDF** (zákon o zájazdoch, nariadenia EÚ o právach cestujúcich, poistné a obchodné podmienky CK)
- **ontológiu „Cestovná kancelária“** z WebProtégé (62 tried, 10 vzťahov)
- **lokálny LLM** cez Ollamu – nič sa neposiela na internet, netreba API kľúč

## Rýchly štart (Windows/macOS/Linux, bez Dockeru) – odporúčané

1. Nainštaluj **Python 3.10+** a **Ollamu** (https://ollama.com/download).
2. Stiahni modely (raz, spolu ~4 GB):
   ```
   ollama pull gemma3:4b
   ollama pull bge-m3
   ```
3. V priečinku projektu:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   python download_pdfs.py        # stiahne PDF do data/pdf/
   python ingest.py               # zaindexuje PDF (prvýkrát niekoľko minút)
   streamlit run app.py           # otvorí appku na http://localhost:8501
   ```

Ak `download_pdfs.py` niektoré PDF nestiahne (EUR-Lex občas blokuje skripty), vypíše odkaz –
otvor ho v prehliadači a PDF ulož do `data/pdf/` pod uvedeným názvom. PDF sa dajú pridať aj
priamo v appke (záložka **Dokumenty → Pridať PDF → Indexovať nové PDF**).

## Spustenie cez Docker (všetko jedným príkazom)

```
python download_pdfs.py
docker compose up --build
```
Prvé spustenie stiahne modely do Docker volume. Potom otvor http://localhost:8501,
v záložke **Dokumenty** klikni **Indexovať nové PDF**. V Docker Desktope nastav aspoň 8 GB RAM.

## Štruktúra

| súbor | úloha |
|---|---|
| `config.py` | nastavenia (modely, veľkosť chunkov, top-k, bonus ontológie) |
| `ontology.py` | načítanie OWL (rdflib), rozpoznanie pojmov, SPARQL, fakty, rozšírenie otázky |
| `ingest.py` | PDF → text → chunky → anotácia pojmami → embeddingy → ChromaDB |
| `rag.py` | hybridné vyhľadávanie (3 režimy), skladanie promptu, generovanie odpovede |
| `lexical.py` | vyhľadávanie podľa kľúčových slov (BM25) so skracovaním slovenských slov |
| `app.py` | Streamlit aplikácia (chat, porovnanie režimov, ontológia, dokumenty) |
| `llm.py` | klient pre Ollama API |
| `download_pdfs.py` | stiahnutie doménových PDF |
| `evaluate.py`, `eval_questions.json` | vyhodnotenie: bez RAG vs. PDF vs. PDF + ontológia |
| `data/ontology/cestovna-kancelaria.owl` | ontológia z WebProtégé |
| `data/ontology/synonyma.json` | doplnkové synonymá k triedam (lietadlo → Letecká doprava) |

## Ako je napojená ontológia

1. **Rozpoznanie pojmov** – v otázke aj v každom úryvku PDF sa hľadajú triedy ontológie podľa
   `rdfs:label` (so slovenským skloňovaním: *Grécko / v Grécku*, *Egypt / do Egypta*) a podľa synoným.
2. **Fakty do promptu** – pre nájdené pojmy sa z ontológie vytiahne hierarchia (nadtriedy, podtriedy,
   alternatívy) a vzťahy; LLM ich dostane ako sekciu `FAKTY Z ONTOLÓGIE [O]`.
3. **Rozšírenie otázky** – *„destinácie v Afrike“* sa hľadá aj ako *Egypt, Keňa, Tunisko*.
4. **Ontologické preradenie** – úryvky, ktoré spomínajú pojem z otázky alebo jeho podtriedu, dostanú bonus ku skóre.

## Zmena modelu

Pomalé? Použi menší model. Lepšia slovenčina? Väčší model (potrebuje viac RAM):
```
set LLM_MODEL=qwen2.5:7b        # Windows (macOS/Linux: export LLM_MODEL=qwen2.5:7b)
ollama pull qwen2.5:7b
streamlit run app.py
```
Po zmene `EMBED_MODEL` treba index postaviť odznova: `python ingest.py --reset`.

## Vyhodnotenie

```
python evaluate.py --no-llm    # rýchlo: iba úspešnosť vyhľadávania
python evaluate.py             # aj s odpoveďami vo všetkých 3 režimoch → eval_vysledky.md
```
