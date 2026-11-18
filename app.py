"""
Streamlit aplikácia – RAG asistent cestovnej kancelárie napojený na ontológiu.
Spustenie:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from ingest import index_stats, ingest
from llm import OllamaClient
from ontology import load_ontology
from rag import MODES, RAG, Retrieval, clean_citations

st.set_page_config(page_title="RAG – Cestovná kancelária", page_icon="", layout="wide")


@st.cache_resource
def get_rag() -> RAG:
    return RAG()


def reset_rag():
    get_rag.clear()


onto = load_ontology()
client = OllamaClient()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("RAG – Cestovná kancelária")
    st.caption("Retrieval-Augmented Generation nad PDF dokumentmi + ontológia z WebProtégé")
    ok, msg = client.check()
    (st.success if ok else st.error)(msg)

    mode = st.radio("Režim odpovedania", list(MODES), format_func=MODES.get, index=0)
    top_k = st.slider("Počet úryvkov v kontexte (top-k)", 1, 10, config.TOP_K)
    show_prompt = st.checkbox("Zobraziť celý prompt", value=False)

    stats = index_stats()
    st.divider()
    st.metric("Úryvkov v indexe", stats["chunks"])
    st.metric("PDF dokumentov", len(stats["sources"]))
    st.metric("Tried v ontológii", len(onto.concepts))
    if st.button("Vymazať konverzáciu"):
        st.session_state.history = []
        st.rerun()


def render_retrieval(r: Retrieval, key: str):
    """Zobrazí, odkiaľ odpoveď pochádza – pojmy z ontológie a citované úryvky."""
    if r.onto is not None:
        with st.expander(f"Ontológia – rozpoznané pojmy: {', '.join(r.onto.concepts) or 'žiadne'}"):
            if r.onto.facts:
                st.markdown("\n".join(f"- {f}" for f in r.onto.facts))
            if r.expanded_query:
                st.caption("Rozšírená otázka pre vyhľadávanie:")
                st.code(r.expanded_query, language=None)
            for i, q in enumerate(r.onto.sparql):
                st.caption("SPARQL dotaz nad ontológiou:")
                st.code(q, language="sparql")
    if r.hits:
        with st.expander(f"Zdroje ({len(r.hits)} úryvkov)"):
            for i, h in enumerate(r.hits, 1):
                bonus = f" {'+' if h.onto_bonus > 0 else '−'} ontológia {abs(h.onto_bonus):.2f}" if h.onto_bonus else ""
                lex = f" + kľúčové slová {h.lexical:.2f}" if h.lexical >= 0.005 else ""
                st.markdown(f"**[{i}] {h.source}, strana {h.page}** — podobnosť {h.similarity:.3f}{lex}{bonus}"
                            + (f"  \n_pojmy:_ {', '.join(h.concepts)}" if h.concepts else ""))
                st.text(h.text[:1200])
    if show_prompt:
        with st.expander("Prompt poslaný do LLM"):
            for m in r.messages:
                st.markdown(f"**{m['role']}**")
                st.text(m["content"])


tab_chat, tab_cmp, tab_onto, tab_docs = st.tabs(
    ["Chat", "Porovnanie režimov", "Ontológia", "Dokumenty"])

# ---------------------------------------------------------------- chat
with tab_chat:
    st.session_state.setdefault("history", [])
    messages_box = st.container()          # správy sa kreslia sem, nad políčko na písanie
    question = st.chat_input("Opýtaj sa niečo o zájazdoch, právach cestujúcich, poistení…")

    with messages_box:
        if not st.session_state.history and not question:
            st.info("Príklady otázok: *Na čo mám nárok, ak mi meškal let do Egypta?* · "
                    "*Aké typy pobytových zájazdov ponúkate?* · "
                    "*Do kedy môže cestovná kancelária zrušiť zájazd?* · "
                    "*Naplánuj mi týždennú dovolenku pri mori v lete s deťmi.*")
        for i, msg_ in enumerate(st.session_state.history):
            with st.chat_message(msg_["role"]):
                st.markdown(msg_["content"])
                if msg_.get("retrieval"):
                    st.caption(MODES[msg_["retrieval"].mode])
                    render_retrieval(msg_["retrieval"], key=f"h{i}")

        if question:
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                if not ok:
                    st.error(msg)
                else:
                    with st.spinner("Hľadám v dokumentoch a ontológii…"):
                        stream, r = get_rag().answer_stream(question, mode, top_k)
                    placeholder, answer = st.empty(), ""
                    for token in stream:
                        answer += token
                        placeholder.markdown(answer + "▌")
                    answer = clean_citations(answer, len(r.hits))
                    placeholder.markdown(answer)
                    st.caption(MODES[mode])
                    render_retrieval(r, key="new")
                    st.session_state.history.append({"role": "assistant", "content": answer, "retrieval": r})

# ---------------------------------------------------------------- porovnanie
with tab_cmp:
    st.subheader("Rovnaká otázka v troch režimoch")
    st.caption("Ukazuje prínos RAG a ontológie: bez RAG model háda, s PDF cituje dokumenty, "
               "s ontológiou pozná aj ponuku a pojmy cestovnej kancelárie.")
    q2 = st.text_input("Otázka", "Aké typy pobytových zájazdov ponúkate a čo ak mi pri lete do Egypta zrušia let?")
    if st.button("Porovnať", type="primary", disabled=not ok):
        cols = st.columns(3)
        for col, m in zip(cols, ["llm", "pdf", "pdf+onto"]):
            with col:
                st.markdown(f"#### {MODES[m]}")
                with st.spinner("Generujem…"):
                    text, r = get_rag().answer(q2, m, top_k)
                st.markdown(text)
                render_retrieval(r, key=f"cmp-{m}")

# ---------------------------------------------------------------- ontológia
with tab_onto:
    s = onto.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Triedy", s["triedy"])
    c2.metric("Vzťahy (properties)", s["vztahy"])
    c3.metric("RDF trojice", s["trojice"])

    test_q = st.text_input("Vyskúšaj rozpoznanie pojmov v texte", "Chcem all inclusive na Kréte a letenku z Bratislavy")
    ctx = onto.context_for_query(test_q)
    st.write("**Rozpoznané triedy:**", ", ".join(ctx.concepts) or "—")
    st.write("**Rozšírenie otázky:**", ", ".join(ctx.expanded_terms) or "—")
    if ctx.facts:
        st.markdown("\n".join(f"- {f}" for f in ctx.facts))

    st.markdown("#### Hierarchia tried (zvýraznené = rozpoznané pojmy)")
    st.graphviz_chart(onto.to_dot(set(ctx.concepts)), width="stretch")

    st.markdown("#### Vzťahy (object/data properties)")
    st.dataframe(pd.DataFrame([{"vzťah": r.label, "typ": r.kind, "doména": ", ".join(r.domain),
                                "rozsah": ", ".join(r.range)} for r in onto.relations]),
                 width="stretch", hide_index=True)

    st.markdown("#### SPARQL nad ontológiou")
    default_q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?trieda ?nadtrieda WHERE {
  ?t rdfs:subClassOf ?n .
  ?t rdfs:label ?trieda .
  ?n rdfs:label ?nadtrieda .
} ORDER BY ?nadtrieda ?trieda"""
    sparql = st.text_area("Dotaz", default_q, height=160)
    if st.button("Spustiť SPARQL"):
        try:
            rows = onto.run_sparql(sparql)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Chyba v dotaze: {e}")

# ---------------------------------------------------------------- dokumenty
with tab_docs:
    st.subheader("Znalostná báza (PDF)")
    if stats["sources"]:
        st.dataframe(pd.DataFrame([{"dokument": k, "úryvkov": v} for k, v in stats["sources"].items()]),
                     width="stretch", hide_index=True)
    else:
        st.warning("Index je prázdny. Nahraj PDF a klikni na Indexovať.")

    pdfs = sorted(p.name for p in config.PDF_DIR.glob("*.pdf"))
    not_indexed = [p for p in pdfs if p not in stats["sources"]]
    st.caption(f"PDF v priečinku `{config.PDF_DIR.name}/`: {len(pdfs)} · neindexované: {len(not_indexed)}")

    uploaded = st.file_uploader("Pridať PDF", type="pdf", accept_multiple_files=True)
    if uploaded:
        config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (config.PDF_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Uložené: {', '.join(f.name for f in uploaded)} – teraz klikni na Indexovať.")

    c1, c2 = st.columns(2)
    run_new = c1.button("Indexovať nové PDF", type="primary", disabled=not ok)
    run_all = c2.button("Prebudovať celý index", disabled=not ok)
    if run_new or run_all:
        bar = st.progress(0.0, "Začínam…")
        try:
            rep = ingest(reset=run_all, client=client,
                         progress=lambda t, p: bar.progress(min(max(p, 0.0), 1.0), t))
            bar.progress(1.0, "Hotovo")
            st.success(f"Zaindexovaných PDF: {rep['pdf']}, nových úryvkov: {rep['chunks']}, "
                       f"spolu v indexe: {rep['total_chunks']} ({rep['seconds']} s)")
            if rep["empty"]:
                st.warning("Bez textu (skenované?): " + ", ".join(rep["empty"]))
            reset_rag()
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
