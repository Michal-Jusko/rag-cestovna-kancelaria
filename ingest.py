"""
Indexovanie PDF dokumentov:  PDF → text po stranách → chunky → anotácia pojmami z ontológie
→ embeddingy (Ollama) → ChromaDB.

Spustenie:  python ingest.py            (pridá iba nové PDF)
            python ingest.py --reset    (zmaže index a postaví ho celý odznova)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from llm import OllamaClient
from ontology import load_ontology


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:  # noqa: BLE001 – kolekcia ešte neexistuje
            pass
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def clean_text(text: str) -> str:
    text = text.replace("­", "")                  # mäkké rozdeľovníky
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # rozdelené slová na konci riadku
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Vráti zoznam (číslo strany, text)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = clean_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            txt = ""
        if len(txt) > 30:
            pages.append((i, txt))
    return pages


def chunk_pdf(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    chunks = []
    for page_no, text in load_pdf(path):
        for j, piece in enumerate(splitter.split_text(text)):
            uid = hashlib.md5(f"{path.name}|{page_no}|{j}|{piece[:50]}".encode()).hexdigest()
            chunks.append({"id": uid, "text": piece,
                           "meta": {"source": path.name, "page": page_no, "chunk": j}})
    return chunks


def ingest(reset: bool = False, client: OllamaClient | None = None, batch_size: int = 16,
           progress=None) -> dict:
    """Zaindexuje všetky PDF z config.PDF_DIR. `progress(text, podiel)` je voliteľný callback pre UI."""
    client = client or OllamaClient()
    onto = load_ontology()
    col = get_collection(reset)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""])

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"V priečinku {config.PDF_DIR} nie sú žiadne PDF. "
                                f"Spusti `python download_pdfs.py` alebo ich tam nakopíruj.")

    existing = set()
    if not reset and col.count():
        existing = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}

    report = {"pdf": 0, "chunks": 0, "skipped": [], "empty": []}
    t0 = time.time()
    for n, pdf in enumerate(pdfs, start=1):
        if pdf.name in existing:
            report["skipped"].append(pdf.name)
            continue
        chunks = chunk_pdf(pdf, splitter)
        if not chunks:
            report["empty"].append(pdf.name)   # pravdepodobne skenované PDF bez textovej vrstvy
            print(f"[!] {pdf.name}: nenašiel sa text (skenované PDF?) – preskakujem")
            continue
        for c in chunks:                        # anotácia pojmami z ontológie
            c["meta"].update(onto.annotate(c["text"]))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = client.embed([c["text"] for c in batch])
            col.upsert(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                       embeddings=vectors, metadatas=[c["meta"] for c in batch])
            if progress:
                progress(f"{pdf.name}: {min(i + batch_size, len(chunks))}/{len(chunks)} úryvkov",
                         (n - 1 + (i + batch_size) / len(chunks)) / len(pdfs))
        report["pdf"] += 1
        report["chunks"] += len(chunks)
        print(f"[+] {pdf.name}: {len(chunks)} úryvkov")

    report["total_chunks"] = col.count()
    report["seconds"] = round(time.time() - t0, 1)
    return report


def index_stats() -> dict:
    col = get_collection()
    if not col.count():
        return {"chunks": 0, "sources": {}}
    metas = col.get(include=["metadatas"])["metadatas"]
    sources: dict[str, int] = {}
    for m in metas:
        sources[m["source"]] = sources.get(m["source"], 0) + 1
    return {"chunks": len(metas), "sources": dict(sorted(sources.items()))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Indexovanie PDF do ChromaDB")
    ap.add_argument("--reset", action="store_true", help="zmazať index a postaviť ho odznova")
    args = ap.parse_args()
    ok, msg = OllamaClient().check()
    print(msg)
    if not ok:
        raise SystemExit(1)
    print(ingest(reset=args.reset))
