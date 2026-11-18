"""
RAG pipeline: Retrieve → Augment → Generate, v troch režimoch:

  "llm"       – iba jazykový model, bez dokumentov (na porovnanie)
  "pdf"       – klasický RAG: vektorové hľadanie v PDF
  "pdf+onto"  – RAG napojený na ontológiu: rozpoznanie pojmov, rozšírenie otázky,
                ontologické preradenie úryvkov a fakty z ontológie v prompte
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

import config
from ingest import get_collection
from llm import OllamaClient
from lexical import BM25
from ontology import OntologyContext, load_ontology

MODES = {
    "pdf+onto": "RAG + ontológia",
    "pdf": "RAG (iba PDF)",
    "llm": "Bez RAG (iba LLM)",
}

SYSTEM_PROMPT = """Si asistent cestovnej kancelárie. Odpovedáš po slovensky, vecne a stručne.
Odpovedaj na otázku pomocou kontextu nižšie (FAKTY Z ONTOLÓGIE a ÚRYVKY Z DOKUMENTOV).
- Použi všetko, čo sa k otázke v kontexte nachádza, aj keď je odpoveď iba čiastočná. Uveď konkrétne čísla (sumy, lehoty).
- Ak sa v kontexte k otázke nenachádza nič, povedz, že to v dokumentoch nie je. Nič si nevymýšľaj.
- Fakty môžeš logicky spojiť (napr. Thajsko patrí do Ázie, teda je mimo Európy).
- Úryvky o inej téme ignoruj (napr. o inom druhu dopravy, než sa pýta otázka).
- Za vetu z dokumentov daj zdroj [1] až [N]. Čísla ako (19) vnútri úryvkov nie sú zdroje. Za fakt z ontológie daj [O]."""

# pridáva sa IBA pri otázkach na výber/plánovanie – inak by malý model ponúkal zájazdy pri každej otázke
PLANNING_RULE = """
- Používateľ vyberá alebo plánuje dovolenku: odporuč 1 až 3 zájazdy, ktoré sú v úryvkoch z katalógu
  (kód, názov, termín, noci, cena), a ku každému jednou vetou zdôvodni výber.
  Odporúčaj IBA zájazdy, ktoré sú v úryvkoch doslova uvedené – kódy K01 až K13. Nevymýšľaj iné."""

_PLANNING = re.compile(
    r"napl[aá]nuj|pl[aá]nuj|odpor[uú][čc]|vyber|vybra[tť]|porad|kam (ís[tť]|cestova[tť]|vyrazi[tť])|"
    r"chcem[e]? (ís[tť]|cestova[tť]|dovolenk)|dovolenk|ak[ýy] z[aá]jazd|m[aá]te (nie[čc]o|z[aá]jazd)|hľad[aá]m", re.I)


def is_planning(question: str) -> bool:
    return bool(_PLANNING.search(question))

SYSTEM_PROMPT_NO_RAG = "Si asistent cestovnej kancelárie. Odpovedáš po slovensky, vecne a stručne."


_CIT = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")


def clean_citations(text: str, n_sources: int) -> str:
    """
    Odstráni citácie na neexistujúce zdroje. Model niekedy prepíše číslovanie bodov z textu
    nariadenia, napr. „(19)“, na citáciu [19], hoci dostal iba 5 úryvkov.
    """
    def fix(m):
        nums = [x for x in re.split(r"[,;\s]+", m.group(1)) if x]
        ok = [x for x in nums if 1 <= int(x) <= n_sources]
        return "[" + ", ".join(ok) + "]" if ok else ""
    text = _CIT.sub(fix, text)
    # zvyšky po odstránených citáciách: „Podľa [1] a by…“ → „Podľa [1] by…“, „podľa by ste“ → „by ste“
    text = re.sub(r"(\[[\d, ]+\])\s+(?:a|aj|alebo)\s+(?=[a-záäčďéíĺľňóôŕšťúýž.,])", r"\1 ", text)
    text = re.sub(r"\b([Pp])odľa\s+(?=(?:by|sa|je|sú|má|mal|mali|musí|treba)\b)", "", text)
    text = re.sub(r"(?<=[.!?]\s)([a-záäčďéíĺľňóôŕšťúýž])", lambda m: m.group(1).upper(), text)
    # úvodné „nenašiel som“, ak za ním nasleduje skutočná odpoveď
    m = re.match(r"\s*(V dostupných dokumentoch[^.\n]*nenašiel[^.\n]*\.)\s*", text)
    if m and len(text) > m.end() + 80:
        text = text[m.end():]
    text = re.sub(r"(\[[\d, ]+\])(\s*\1)+", r"\1", text)   # [1] [1] → [1]
    text = re.sub(r"[ \t]+([.,;])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


@dataclass
class Hit:
    text: str
    source: str
    page: int
    similarity: float          # kosínusová podobnosť z vektorového hľadania
    score: float               # výsledné skóre (podobnosť + kľúčové slová + ontológia)
    lexical: float = 0.0       # príspevok zhody kľúčových slov (BM25)
    concepts: list[str] = field(default_factory=list)
    onto_bonus: float = 0.0


@dataclass
class Retrieval:
    mode: str
    question: str
    hits: list[Hit]
    onto: OntologyContext | None
    expanded_query: str | None
    messages: list[dict]


class RAG:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.onto = load_ontology()
        self.col = get_collection()
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Načíta všetky úryvky a ich vektory z ChromaDB do pamäte (pre hybridné vyhľadávanie)."""
        self.docs, self.metas = [], []
        if self.col.count() == 0:
            return
        data = self.col.get(include=["documents", "metadatas", "embeddings"])
        self.docs, self.metas = data["documents"], data["metadatas"]
        E = np.asarray(data["embeddings"], dtype=np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        self.bm25 = BM25(self.docs)

    # ------------------------------------------------------------------ retrieve
    def _hybrid_search(self, queries: list[str], lexical_query: str, n: int) -> list[Hit]:
        """
        Hybridné vyhľadávanie: kosínusová podobnosť embeddingov (význam)
        + BM25 zhoda kľúčových slov (presné výrazy). Pri viacerých otázkach
        (pôvodná + rozšírená ontológiou) sa berie najvyššia podobnosť.
        """
        if not self.docs:
            return []
        Q = np.asarray(self.client.embed(queries), dtype=np.float32)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        sims = (self.E @ Q.T).max(axis=1)
        lex = self.bm25.score(lexical_query)
        if lex.max() > 0:
            lex = lex / lex.max()
        lex = config.LEXICAL_WEIGHT * lex
        combined = sims + lex
        hits = []
        for i in np.argsort(-combined)[:n]:
            m = self.metas[i]
            hits.append(Hit(text=self.docs[i], source=m["source"], page=int(m["page"]),
                            similarity=float(sims[i]), lexical=float(lex[i]),
                            score=float(combined[i])))
        return hits

    def retrieve(self, question: str, mode: str = "pdf+onto", top_k: int = config.TOP_K) -> Retrieval:
        if mode == "llm":
            return Retrieval(mode, question, [], None, None, self._messages(question, [], None, mode))

        onto_ctx, expanded, queries = None, None, [question]
        if mode == "pdf+onto":
            onto_ctx = self.onto.context_for_query(question)
            if onto_ctx.concepts:
                # rozšírená otázka: pôvodná + pojmy z ontológie
                expanded = f"{question} ({', '.join(onto_ctx.concepts + onto_ctx.expanded_terms[:8])})"
                queries.append(expanded)

        hits = self._hybrid_search(queries, question, config.CANDIDATES)

        if onto_ctx is not None:
            # ontologické preradenie:
            #  + bonus, ak úryvok spomína pojem z otázky alebo jeho podtriedu
            #  - penalizácia, ak spomína iba alternatívu (napr. autobus pri otázke o lete)
            for h in hits:
                h.concepts = [c.label for c in self.onto.find_concepts(h.text)]
                found = set(h.concepts)
                overlap = len(found & onto_ctx.boost_set)
                h.onto_bonus = config.ONTOLOGY_BOOST * min(overlap, 2)
                if found & onto_ctx.conflict_set and not overlap:
                    h.onto_bonus -= config.ONTOLOGY_PENALTY
                elif found & onto_ctx.conflict_set:
                    h.onto_bonus -= config.ONTOLOGY_PENALTY / 2
                h.score = h.similarity + h.lexical + h.onto_bonus
            hits.sort(key=lambda h: h.score, reverse=True)

        hits = [h for h in hits if h.similarity >= config.MIN_SCORE][:top_k]
        return Retrieval(mode, question, hits, onto_ctx, expanded,
                         self._messages(question, hits, onto_ctx, mode))

    # ------------------------------------------------------------------ augment
    @staticmethod
    def _messages(question: str, hits: list[Hit], onto_ctx: OntologyContext | None,
                  mode: str) -> list[dict]:
        if mode == "llm":
            return [{"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
                    {"role": "user", "content": question}]
        parts = []
        if onto_ctx is not None:
            facts = "\n".join(f"- {f}" for f in onto_ctx.facts) or "- (v otázke sa nenašiel žiadny pojem z ontológie)"
            parts.append(f"FAKTY Z ONTOLÓGIE CESTOVNEJ KANCELÁRIE [O]:\n{facts}")
        docs = "\n\n".join(f"[{i}] (zdroj: {h.source}, strana {h.page})\n{h.text}"
                           for i, h in enumerate(hits, start=1)) or "(nenašli sa žiadne relevantné úryvky)"
        parts.append(f"ÚRYVKY Z DOKUMENTOV:\n{docs}")
        parts.append(f"OTÁZKA: {question}")
        system = SYSTEM_PROMPT + (PLANNING_RULE if is_planning(question) else "")
        return [{"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(parts)}]

    # ------------------------------------------------------------------ generate
    def answer(self, question: str, mode: str = "pdf+onto", top_k: int = config.TOP_K) -> tuple[str, Retrieval]:
        r = self.retrieve(question, mode, top_k)
        return clean_citations(self.client.chat(r.messages), len(r.hits)), r

    def answer_stream(self, question: str, mode: str = "pdf+onto",
                      top_k: int = config.TOP_K) -> tuple[Iterator[str], Retrieval]:
        r = self.retrieve(question, mode, top_k)
        return self.client.chat_stream(r.messages), r


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Na čo mám nárok, ak mi meškal let do Egypta?"
    rag = RAG()
    text, r = rag.answer(q)
    print("Pojmy z ontológie:", r.onto.concepts if r.onto else "-")
    for i, h in enumerate(r.hits, 1):
        print(f"[{i}] {h.source} s.{h.page}  podobnosť={h.similarity:.3f}  text={h.lexical:.2f}  ontológia={h.onto_bonus:+.2f}")
    print("\n" + text)
