"""Centrálna konfigurácia projektu. Hodnoty sa dajú prepísať premennými prostredia."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Vstupné dáta
PDF_DIR = Path(os.getenv("PDF_DIR", DATA_DIR / "pdf"))
ONTOLOGY_PATH = Path(os.getenv("ONTOLOGY_PATH", DATA_DIR / "ontology" / "cestovna-kancelaria.owl"))
SYNONYMS_PATH = Path(os.getenv("SYNONYMS_PATH", DATA_DIR / "ontology" / "synonyma.json"))

# Vektorová databáza (ChromaDB, ukladá sa lokálne na disk)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION = "cestovna_kancelaria"

# Ollama – lokálny LLM a embedding model
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")      # alternatíva: qwen2.5:7b, llama3.1:8b
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")     # viacjazyčný embedding (zvláda slovenčinu)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))       # znakov
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))                   # koľko úryvkov ide do promptu
CANDIDATES = int(os.getenv("CANDIDATES", 20))        # koľko kandidátov vráti vektorové hľadanie
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", 0.15))  # váha zhody kľúčových slov (BM25) v hybridnom hľadaní
ONTOLOGY_BOOST = float(os.getenv("ONTOLOGY_BOOST", 0.06))  # bonus k skóre za zhodu pojmov z ontológie
ONTOLOGY_PENALTY = float(os.getenv("ONTOLOGY_PENALTY", 0.08))  # postih za úryvok o alternatíve (iný druh dopravy...)
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.25))      # úryvky pod týmto skóre sa zahodia
