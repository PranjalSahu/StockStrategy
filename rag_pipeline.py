import yfinance as yf
import feedparser
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Dict

# ==========================
# CONFIG
# ==========================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:1b"#"""gemma2:2b"

TICKERS = open("tickers.txt").readlines()#["AAPL", "MSFT", "NVDA"]
TICKERS = [x.strip() for x in TICKERS]

TOP_K = 5
MAX_CHARS = 800

DB_DIR = Path("rag_db")
META_FILE = DB_DIR / "metadata.json"
EMB_FILE = DB_DIR / "embeddings.npy"

REFRESH_DB = True   # set True to force refetch


# ==========================
# OLLAMA HELPERS
# ==========================
def ollama_embed(texts: List[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return np.array(embeddings)


def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
        },
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ==========================
# DATA FETCHING
# ==========================
def fetch_stock_data(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    info = stock.info

    return f"""
Ticker: {ticker}
Company: {info.get("longName")}
Sector: {info.get("sector")}
Market Cap: {info.get("marketCap")}
PE Ratio: {info.get("trailingPE")}
52W High: {info.get("fiftyTwoWeekHigh")}
52W Low: {info.get("fiftyTwoWeekLow")}
Business Summary:
{info.get("longBusinessSummary")}
""".strip()


def fetch_news_yahoo(ticker: str, max_items=5) -> List[str]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    articles = []
    for e in feed.entries[:max_items]:
        articles.append(f"""
Title: {e.title}
Published: {e.published}
Summary: {e.summary}
""".strip())
    return articles


# ==========================
# UTILS
# ==========================
def chunk_text(text: str, max_chars=800) -> List[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


# ==========================
# PERSISTENT VECTOR STORE
# ==========================
class PersistentVectorStore:
    def __init__(self):
        self.texts: List[Dict] = []
        self.embeddings: np.ndarray = None

    def load(self):
        if META_FILE.exists() and EMB_FILE.exists():
            with open(META_FILE, "r") as f:
                self.texts = json.load(f)
            self.embeddings = np.load(EMB_FILE)
            print(f"Loaded {len(self.texts)} chunks from disk")
            return True
        return False

    def save(self):
        DB_DIR.mkdir(exist_ok=True)
        with open(META_FILE, "w") as f:
            json.dump(self.texts, f, indent=2)
        np.save(EMB_FILE, self.embeddings)

    def add(self, records: List[Dict]):
        texts = [r["text"] for r in records]
        new_embs = ollama_embed(texts)

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.texts.extend(records)

    def has_ticker(self, ticker: str) -> bool:
        return any(r["ticker"] == ticker for r in self.texts)

    def search(self, query: str, k=5):
        q_emb = ollama_embed([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = sims.argsort()[-k:][::-1]
        return [(self.texts[i], sims[i]) for i in idxs]


# ==========================
# BUILD / LOAD INDEX
# ==========================
def build_or_load_index(tickers: List[str]) -> PersistentVectorStore:
    store = PersistentVectorStore()

    if not REFRESH_DB and store.load():
        return store

    for ticker in tickers:
        if store.has_ticker(ticker):
            print(f"Skipping {ticker}, already indexed")
            continue

        print(f"Fetching {ticker} data...")

        records = []

        # Fundamentals
        for chunk in chunk_text(fetch_stock_data(ticker)):
            records.append({
                "ticker": ticker,
                "type": "fundamentals",
                "text": chunk,
            })

        # News
        for article in fetch_news_yahoo(ticker):
            for chunk in chunk_text(article):
                records.append({
                    "ticker": ticker,
                    "type": "news",
                    "text": chunk,
                })

        store.add(records)

    store.save()
    print("DB saved to disk")
    return store


# ==========================
# RAG QUERY
# ==========================
def rag_query(store: PersistentVectorStore, question: str) -> str:
    hits = store.search(question, TOP_K)

    context = "\n\n".join(
        f"[{h['ticker']} | {h['type']}]\n{h['text']}"
        for h, _ in hits
    )

    prompt = f"""
You are a financial research assistant.
Answer using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    return ollama_generate(prompt)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    store = build_or_load_index(TICKERS)

    print("\nRAG ready. Ask questions (type exit to quit)\n")
    #while True:
    q = "Tell me about AAPL stock"#input(">>> ")
    #if q.lower() in ["exit", "quit"]:
    #    break
    print("\n", rag_query(store, q), "\n")
