"""
Vyhodnotenie: rovnaké otázky v troch režimoch (bez RAG / iba PDF / PDF + ontológia).

Meria sa:
  - retrieval hit-rate: či sa medzi top-k úryvkami nachádza očakávaný dokument
  - pokrytie ontológie: či systém rozpoznal očakávané pojmy z ontológie
  - samotné odpovede sa uložia do eval_vysledky.md na ručné posúdenie (správnosť, citácie, halucinácie)

Spustenie:  python evaluate.py            (aj s generovaním odpovedí – trvá dlhšie)
            python evaluate.py --no-llm   (iba retrieval, rýchle)
"""
import argparse
import json
from datetime import datetime

import config
from rag import MODES, RAG, clean_citations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="negenerovať odpovede, merať iba retrieval")
    args = ap.parse_args()

    questions = json.loads((config.BASE_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    rag = RAG()
    rows, out = [], [f"# Vyhodnotenie RAG ({datetime.now():%Y-%m-%d %H:%M})\n",
                     f"LLM: `{rag.client.llm_model}`, embeddingy: `{rag.client.embed_model}`, top-k: {config.TOP_K}\n"]
    hits_pdf = hits_onto = onto_ok = n_ret = 0

    for i, q in enumerate(questions, 1):
        out.append(f"\n## {i}. {q['otazka']}\n")
        out.append(f"*Očakávaný zdroj:* {', '.join(q['zdroj']) or 'ontológia'} · "
                   f"*očakávané pojmy:* {', '.join(q['pojmy']) or '—'}\n")
        for mode in (["llm", "pdf", "pdf+onto"] if not args.no_llm else ["pdf", "pdf+onto"]):
            r = rag.retrieve(q["otazka"], mode)
            sources = [h.source for h in r.hits]
            hit = any(any(z in s for s in sources) for z in q["zdroj"]) if q["zdroj"] else None
            line = f"\n**{MODES[mode]}**"
            if mode != "llm":
                line += f" — zdroje: {', '.join(f'{s} (s.{h.page})' for s, h in zip(sources, r.hits)) or '—'}"
                if hit is not None:
                    line += f" — zásah: {'✅' if hit else '❌'}"
            if mode == "pdf+onto":
                found = set(r.onto.concepts)
                ok = set(q["pojmy"]) <= found
                onto_ok += ok
                line += f" — pojmy: {', '.join(sorted(found)) or '—'} {'✅' if ok else '❌'}"
            out.append(line + "\n")
            if q["zdroj"] and mode == "pdf":
                hits_pdf += bool(hit)
                n_ret += 1
            if q["zdroj"] and mode == "pdf+onto":
                hits_onto += bool(hit)
            if not args.no_llm:
                answer = clean_citations(rag.client.chat(r.messages), len(r.hits))
                out.append("\n> " + answer.replace("\n", "\n> ") + "\n")
        print(f"{i}/{len(questions)} hotovo")

    summary = (f"\n## Súhrn\n\n| metrika | iba PDF | PDF + ontológia |\n|---|---|---|\n"
               f"| zásah očakávaného dokumentu v top-{config.TOP_K} | {hits_pdf}/{n_ret} | {hits_onto}/{n_ret} |\n"
               f"| rozpoznané očakávané pojmy z ontológie | — | {onto_ok}/{len(questions)} |\n")
    out.insert(2, summary)
    (config.BASE_DIR / "eval_vysledky.md").write_text("".join(out), encoding="utf-8")
    print(summary)
    print("Výsledky uložené v eval_vysledky.md")


if __name__ == "__main__":
    main()
