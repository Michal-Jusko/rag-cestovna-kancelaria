"""
Napojenie RAG na ontológiu z WebProtégé (OWL/RDF-XML).

Ontológia sa v RAG používa na štyri veci:
  1. ROZPOZNANIE POJMOV – v otázke aj v každom PDF úryvku sa nájdu triedy ontológie
     (podľa rdfs:label, so zohľadnením slovenského skloňovania + synonymá).
  2. FAKTY DO PROMPTU – pre nájdené pojmy sa SPARQL dotazmi vytiahne hierarchia
     (nadtriedy, podtriedy, súrodenci) a vzťahy (object properties s doménou/rozsahom).
  3. ROZŠÍRENIE OTÁZKY – otázka sa doplní o príbuzné pojmy (napr. „Afrika“ → Egypt, Keňa, Tunisko).
  4. ONTOLOGICKÉ PRERADENIE – úryvky z PDF, ktoré obsahujú pojem z otázky alebo jeho
     podtriedu, dostanú bonus ku skóre podobnosti.

Pozn.: WebProtégé generuje náhodné IRI (napr. R7ApBimHPZfvAaGtqaBYLjN), preto sa všetko
páruje cez rdfs:label, nie cez IRI.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

import config

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# slová, ktoré sa pri párovaní viacslovných názvov ignorujú („Pobyt pri mori“)
# hierarchie, ktorých podtriedy sa navzájom KOMBINUJÚ (rodina = dospelí + deti, viac doplnkových služieb),
# a preto pre ne neplatí penalizácia „alternatív“ pri vyhľadávaní
COMBINABLE_ROOTS = {"Osoba", "Doplnková služba"}

STOPWORDS = {"a", "o", "v", "vo", "s", "so", "z", "zo", "k", "ku", "na", "do", "za", "pri", "pre", "po", "od"}


# ---------------------------------------------------------------------------
# Pomocné funkcie na prácu so slovenčinou
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Malé písmená, bez diakritiky, podčiarkovníky na medzery, iba písmená a čísla."""
    text = unicodedata.normalize("NFKD", text.replace("_", " ").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def stem_match(label_token: str, word: str) -> bool:
    """
    Jednoduché párovanie so skloňovaním: Grécko ~ Grécka/Grécku, Egypt ~ Egypta,
    Letecká doprava ~ leteckou dopravou. Krátke tokeny (USA) sa musia zhodovať presne.
    """
    if label_token == word:
        return True
    n = len(label_token)
    if n <= 3:
        return False
    need = n - 1
    return len(word) >= need and word[:need] == label_token[:need]


# ---------------------------------------------------------------------------
# Dátové štruktúry
# ---------------------------------------------------------------------------
@dataclass
class Concept:
    iri: URIRef
    label: str
    parents: list[URIRef] = field(default_factory=list)
    children: list[URIRef] = field(default_factory=list)
    alt_labels: list[str] = field(default_factory=list)   # skos:altLabel / ďalšie rdfs:label
    comment: str = ""

    @property
    def tokens(self) -> list[str]:
        return [t for t in tokenize(self.label) if t not in STOPWORDS]


@dataclass
class Relation:
    label: str
    domain: list[str]
    range: list[str]
    kind: str  # "object" | "data"


@dataclass
class OntologyContext:
    """Výsledok spracovania otázky ontológiou – ide do retrievalu aj do promptu."""
    concepts: list[str]               # rozpoznané triedy (názvy)
    expanded_terms: list[str]         # príbuzné pojmy na rozšírenie otázky
    boost_set: set[str]               # pojmy, za ktoré dostane úryvok bonus
    facts: list[str]                  # fakty v prirodzenom jazyku pre LLM
    conflict_set: set[str] = field(default_factory=set)  # alternatívy/disjunktné pojmy → penalizácia
    sparql: list[str] = field(default_factory=list)   # použité SPARQL dotazy (na ukážku v UI)


# ---------------------------------------------------------------------------
# Ontológia
# ---------------------------------------------------------------------------
class Ontology:
    def __init__(self, owl_path: Path = config.ONTOLOGY_PATH,
                 synonyms_path: Path | None = config.SYNONYMS_PATH):
        self.owl_path = Path(owl_path)
        self.g = Graph()
        self.g.parse(str(self.owl_path))
        self.concepts: dict[URIRef, Concept] = {}
        self.by_label: dict[str, Concept] = {}
        self.relations: list[Relation] = []
        self.synonyms: dict[str, list[str]] = {}
        self._load_classes()
        self._load_relations()
        self.disjoint: dict[URIRef, set[URIRef]] = {}
        self._load_disjoint()
        if synonyms_path and Path(synonyms_path).exists():
            self.synonyms = json.loads(Path(synonyms_path).read_text(encoding="utf-8"))

    # --- načítanie --------------------------------------------------------
    def _label(self, node) -> str:
        lab = self.g.value(node, RDFS.label)
        if lab is not None:
            return str(lab)
        s = str(node)
        return s.split("#")[-1].split("/")[-1]

    def _load_classes(self) -> None:
        q = """
        SELECT ?c ?label WHERE {
            ?c a owl:Class .
            OPTIONAL { ?c rdfs:label ?label }
            FILTER(isIRI(?c))
        }"""
        for c, _ in self.g.query(q, initNs={"owl": OWL, "rdfs": RDFS}):
            if c in self.concepts:
                continue
            labels = [str(x) for x in self.g.objects(c, RDFS.label)]
            main = labels[0] if labels else self._label(c)
            alts = labels[1:] + [str(x) for x in self.g.objects(c, SKOS.altLabel)]
            comment = " ".join(str(x) for x in self.g.objects(c, RDFS.comment))
            self.concepts[c] = Concept(iri=c, label=main.replace("_", " ").strip(),
                                       alt_labels=alts, comment=comment)
        for c in self.concepts:
            for p in self.g.objects(c, RDFS.subClassOf):
                if p in self.concepts:
                    self.concepts[c].parents.append(p)
                    self.concepts[p].children.append(c)
        for c in self.concepts.values():
            c.children.sort(key=lambda x: self.concepts[x].label)
            self.by_label[c.label] = c

    def _load_relations(self) -> None:
        for kind, t in (("object", OWL.ObjectProperty), ("data", OWL.DatatypeProperty)):
            for p in self.g.subjects(RDF.type, t):
                self.relations.append(Relation(
                    label=self._label(p),
                    domain=[self._label(d) for d in self.g.objects(p, RDFS.domain)],
                    range=[self._label(r) for r in self.g.objects(p, RDFS.range)],
                    kind=kind))
        self.relations.sort(key=lambda r: r.label)

    def _load_disjoint(self) -> None:
        """owl:disjointWith a owl:AllDisjointClasses z ontológie."""
        from rdflib.collection import Collection
        pairs = list(self.g.subject_objects(OWL.disjointWith))
        for d in self.g.subjects(RDF.type, OWL.AllDisjointClasses):
            members = list(Collection(self.g, self.g.value(d, OWL.members)))
            pairs += [(a, b) for a in members for b in members if a != b]
        for a, b in pairs:
            if a in self.concepts and b in self.concepts:
                self.disjoint.setdefault(a, set()).add(b)
                self.disjoint.setdefault(b, set()).add(a)

    # --- navigácia v hierarchii ------------------------------------------
    def get(self, label: str) -> Concept:
        return self.by_label[label]

    def ancestors(self, c: Concept) -> list[Concept]:
        out, stack = [], list(c.parents)
        while stack:
            p = self.concepts[stack.pop()]
            if p not in out:
                out.append(p)
                stack.extend(p.parents)
        return out

    def descendants(self, c: Concept) -> list[Concept]:
        out, stack = [], list(c.children)
        while stack:
            ch = self.concepts[stack.pop()]
            if ch not in out:
                out.append(ch)
                stack.extend(ch.children)
        return out

    def path(self, c: Concept) -> list[str]:
        """Cesta od koreňa: Destinácia → Afrika → Egypt."""
        p, cur = [c.label], c
        while cur.parents:
            cur = self.concepts[cur.parents[0]]
            p.append(cur.label)
        return list(reversed(p))

    def roots(self) -> list[Concept]:
        return sorted((c for c in self.concepts.values() if not c.parents), key=lambda c: c.label)

    # --- rozpoznanie pojmov v texte --------------------------------------
    def find_concepts(self, text: str) -> list[Concept]:
        """Nájde triedy ontológie spomenuté v texte (otázke alebo úryvku z PDF)."""
        words = tokenize(text)
        if not words:
            return []
        found: list[Concept] = []

        for c in self.concepts.values():
            # a) názov triedy (a alternatívne názvy) so skloňovaním
            for label in [c.label, *c.alt_labels]:
                toks = [t for t in tokenize(label) if t not in STOPWORDS]
                if toks and self._label_hit(toks, words):
                    found.append(c)
                    break
            else:
                # b) synonymá zo súboru synonyma.json („lietadlo“ → Letecká doprava)
                for syn in self.synonyms.get(c.label, []):
                    if self._synonym_hit(syn, words):
                        found.append(c)
                        break

        # ak sa našla „Letecká doprava“, netreba zvlášť všeobecnú „Doprava“
        found_set = set(id(c) for c in found)
        result = []
        for c in found:
            more_specific = any(
                id(d) in found_set and d is not c and set(c.tokens) < set(d.tokens)
                and c in self.ancestors(d)
                for d in found)
            if not more_specific:
                result.append(c)
        return result

    @staticmethod
    def _label_hit(toks: list[str], words: list[str]) -> bool:
        """Všetky slová názvu triedy sa musia nájsť blízko seba (okno = počet slov + 2)."""
        if len(toks) == 1:
            return any(stem_match(toks[0], w) for w in words)
        window = len(toks) + 2
        for i, w in enumerate(words):
            if stem_match(toks[0], w):
                span = words[max(0, i - window): i + window + 1]
                if all(any(stem_match(t, x) for x in span) for t in toks[1:]):
                    return True
        return False

    @staticmethod
    def _synonym_hit(syn: str, words: list[str]) -> bool:
        """Synonymum = jedno alebo viac slov; slovo s * na konci je prefix („lietadl*“)."""
        parts = []
        for raw in syn.split():
            is_prefix = raw.endswith("*")
            n = normalize(raw.rstrip("*"))
            if n:
                parts.append((n, is_prefix))
        if not parts:
            return False

        def ok(word, part):
            return word.startswith(part[0]) if part[1] else word == part[0]

        k = len(parts)
        return any(all(ok(words[i + j], parts[j]) for j in range(k))
                   for i in range(len(words) - k + 1))

    # --- fakty pre LLM ----------------------------------------------------
    def describe(self, c: Concept, max_children: int = 25) -> list[str]:
        facts = []
        path = self.path(c)
        if len(path) > 1:
            facts.append(f"„{c.label}“ je v ontológii podtriedou „{path[-2]}“ "
                         f"(hierarchia: {' → '.join(path)}).")
        else:
            facts.append(f"„{c.label}“ je hlavná (koreňová) trieda ontológie.")

        if c.children:
            parts = []
            for ch_iri in c.children[:max_children]:
                ch = self.concepts[ch_iri]
                if ch.children:
                    sub = ", ".join(self.concepts[x].label for x in ch.children)
                    parts.append(f"{ch.label} ({sub})")
                else:
                    parts.append(ch.label)
            facts.append(f"„{c.label}“ má podtriedy: {'; '.join(parts)}.")

        if c.parents:
            parent = self.concepts[c.parents[0]]
            sibs = [self.concepts[s].label for s in parent.children if s != c.iri]
            if sibs:
                facts.append(f"Ďalšie podtriedy „{parent.label}“ (alternatívy k „{c.label}“): "
                             f"{', '.join(sibs)}.")

        lineage = {c.label, *(a.label for a in self.ancestors(c))}
        for r in self.relations:
            if lineage & set(r.domain):
                rng = ", ".join(r.range) or "?"
                facts.append(f"Vzťah „{r.label}“: {', '.join(r.domain)} → {rng} "
                             f"(platí aj pre „{c.label}“).")
            elif lineage & set(r.range):
                facts.append(f"Vzťah „{r.label}“: {', '.join(r.domain) or '?'} → {', '.join(r.range)}.")

        if c.comment:
            facts.append(f"Popis „{c.label}“: {c.comment}")
        return facts

    def conflicts(self, matched: list[Concept]) -> set[str]:
        """
        Pojmy, ktoré sú k pojmom z otázky ALTERNATÍVOU, nie zhodou:
        súrodenecké triedy (Letecká doprava vs. Autobusová doprava, Egypt vs. Tunisko)
        a triedy deklarované v ontológii ako disjunktné – vrátane ich podtried.
        """
        keep = set()
        for c in matched:
            keep.add(c.label)
            keep.update(a.label for a in self.ancestors(c))
            keep.update(d.label for d in self.descendants(c))
        out = set()
        for c in matched:
            # disjunktnosť hlavných kategórií (Destinácia vs. Doprava…) neznamená alternatívu,
            # preto sa berie do úvahy iba pri triedach, ktoré majú nadtriedu
            others = set(self.disjoint.get(c.iri, set())) if c.parents else set()
            root = self.path(c)[0]
            for p in c.parents:
                if root in COMBINABLE_ROOTS:
                    break
                others.update(x for x in self.concepts[p].children if x != c.iri)
            for o in others:
                oc = self.concepts[o]
                out.add(oc.label)
                out.update(d.label for d in self.descendants(oc))
        return out - keep

    # --- hlavný vstup pre RAG --------------------------------------------
    def context_for_query(self, question: str) -> OntologyContext:
        matched = self.find_concepts(question)
        facts, expanded, boost, sparql = [], [], set(), []
        for c in matched:
            facts.extend(self.describe(c))
            boost.add(c.label)
            desc = self.descendants(c)
            boost.update(d.label for d in desc)
            # rozšírenie otázky: podtriedy + priama nadtrieda
            expanded.extend(d.label for d in desc[:10])
            if c.parents:
                expanded.append(self.concepts[c.parents[0]].label)
            sparql.append(self.sparql_for(c))

        # odstránenie duplicít so zachovaním poradia
        seen, uniq_facts = set(), []
        for f in facts:
            if f not in seen:
                seen.add(f)
                uniq_facts.append(f)
        matched_labels = [c.label for c in matched]
        expanded = [e for e in dict.fromkeys(expanded) if e not in matched_labels]
        return OntologyContext(concepts=matched_labels, expanded_terms=expanded,
                               boost_set=boost, facts=uniq_facts, sparql=sparql,
                               conflict_set=self.conflicts(matched))

    def sparql_for(self, c: Concept) -> str:
        """SPARQL dotaz, ktorý vráti nadtriedy a podtriedy pojmu (zobrazuje sa v appke)."""
        raw = self.g.value(c.iri, RDFS.label)
        lit = f'"{raw}"' + (f"@{raw.language}" if getattr(raw, "language", None) else "")
        return f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?vztah ?pojem WHERE {{
  ?c rdfs:label {lit} .
  {{ ?c rdfs:subClassOf+ ?x . BIND("nadtrieda" AS ?vztah) }}
  UNION
  {{ ?x rdfs:subClassOf+ ?c . BIND("podtrieda" AS ?vztah) }}
  ?x rdfs:label ?pojem .
}}"""

    def run_sparql(self, query: str) -> list[tuple]:
        return [tuple(str(x) for x in row) for row in self.g.query(query)]

    # --- anotácia úryvkov pri indexovaní ---------------------------------
    def annotate(self, text: str) -> dict[str, str]:
        """Metadáta pre chunk: priamo spomenuté pojmy a pojmy vrátane nadtried."""
        direct = self.find_concepts(text)
        up = {c.label for c in direct}
        for c in direct:
            up.update(a.label for a in self.ancestors(c))
        return {"concepts": "|" + "|".join(sorted(c.label for c in direct)) + "|",
                "concepts_up": "|" + "|".join(sorted(up)) + "|"}

    # --- export pre appku -------------------------------------------------
    def to_dot(self, highlight: set[str] | None = None) -> str:
        """Graf hierarchie tried v jazyku DOT (Streamlit ho vykreslí)."""
        highlight = highlight or set()
        lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box, style="rounded,filled", '
                 'fillcolor="#f4f4f5", fontname="Helvetica", fontsize=10];']
        for c in self.concepts.values():
            color = "#fde68a" if c.label in highlight else "#f4f4f5"
            lines.append(f'"{c.label}" [fillcolor="{color}"];')
            for p in c.parents:
                lines.append(f'"{self.concepts[p].label}" -> "{c.label}";')
        lines.append("}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {"triedy": len(self.concepts),
                "vztahy": len(self.relations),
                "trojice": len(self.g),
                "korenove_triedy": [c.label for c in self.roots()]}


@lru_cache(maxsize=1)
def load_ontology() -> Ontology:
    return Ontology()


if __name__ == "__main__":
    # rýchly test z príkazového riadka: python ontology.py "Aké stravovanie ponúkate v Egypte?"
    import sys
    onto = load_ontology()
    print(onto.stats())
    q = " ".join(sys.argv[1:]) or "Potrebujem cestovné poistenie na let do Egypta s polpenziou."
    ctx = onto.context_for_query(q)
    print("\nOtázka:", q)
    print("Pojmy:", ctx.concepts)
    print("Rozšírenie:", ctx.expanded_terms)
    print("Fakty:")
    for f in ctx.facts:
        print("  -", f)
