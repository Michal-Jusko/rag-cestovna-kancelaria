"""
Lexikálne (kľúčové slová) vyhľadávanie BM25 – druhá polovica hybridného vyhľadávania.

Embeddingy hľadajú podľa významu, ale v dlhých právnych textoch sú si úryvky významovo podobné
(podobnosti 0,55–0,64). BM25 doplní presnú zhodu dôležitých slov („liečebné náklady“, „náhrada“).

Slovenčina sa skloňuje, preto sa slová skracujú na prvé 4 znaky bez diakritiky:
liečebné / liečebných → „liec“, zrušia / zrušenia → „zrus“, Thajsko / Thajsku → „thaj“.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ontology import normalize

STOPWORDS = set("""
a aby ak ako aky aka ake akeho akej akom akych ale alebo ani aj ano bez bol bola boli bolo by byt
cez ci co do ho i ich ja je jej jeho ju k kam kde kedy ked ktory ktora ktore ktoreho ktorej ku lebo
ma mam mat mi mna moj moze mozem mozu mu my na nad nam nas nie nich niekto no o od on ona oni po pod
podla pre pri preco s sa si so su ta tak tam te ten tento tieto to tom tu ty u uz v vo vam vas za ze
z zo kolko sme ste som ste bude budem
""".split())


def stem(word: str) -> str:
    return word[:max(3, min(len(word) - 1, 4))]


def tokens(text: str) -> list[str]:
    return [stem(w) for w in normalize(text).split() if len(w) >= 3 and w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokens(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if len(docs) else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.tfs), dtype=np.float32)
        q = set(tokens(query))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores[i] = s
        return scores
