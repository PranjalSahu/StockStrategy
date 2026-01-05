import time
import hashlib
import requests
import yfinance as yf
import feedparser
from typing import List
import chromadb

# ==========================
# CONFIG
# ==========================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"#"nomic-embed-text"
LLM_MODEL = "gemma3:1b"

REFRESH_INTERVAL = 60 * 60  # 1 hour
TOP_K = 5
MAX_CHARS = 800

TICKERS = [x.strip() for x in open("tickers.txt").readlines()]

# ==========================
# VECTOR DB (CHROMA) - FIXED
# ==========================
# Use PersistentClient instead of Client for guaranteed persistence
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="stocks_rag_gemma"
)

# ==========================
# OLLAMA HELPERS
# ==========================
def ollama_embed(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        r.raise_for_status()
        embeddings.append(r.json()["embedding"])
    return embeddings


def ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "temperature": 0.2,
            "stream": False
        },
        timeout=120
    )
    r.raise_for_status()
    return r.json()["response"]

# ==========================
# DATA FETCHING
# ==========================
def fetch_stock_data(ticker: str) -> str:
    info = yf.Ticker(ticker).info
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
def chunk_text(text: str, max_chars=800):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# ==========================
# INDEXING
# ==========================
def already_indexed(ticker: str, h: str) -> bool:
    results = collection.get(
        where={
            "$and": [
                {"ticker": ticker},
                {"hash": h}
            ]
        },
        limit=1
    )
    return len(results["ids"]) > 0


def index_documents(ticker: str, doc_type: str, texts: List[str]):
    for idx, text in enumerate(texts):
        h = content_hash(text)
        if already_indexed(ticker, h):
            continue
        embedding = ollama_embed([text])[0]
        doc_id = f"{ticker}_{doc_type}_{h}"
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "ticker": ticker,
                "type": doc_type,
                "hash": h
            }]
        )

# ==========================
# PERIODIC INGESTION LOOP - FIXED
# ==========================
def delete_existing(ticker: str, doc_type: str):
    """Delete existing documents for a ticker and type"""
    results = collection.get(
        where={
            "$and": [
                {"ticker": ticker},
                {"type": doc_type}
            ]
        }
    )
    
    if len(results["ids"]) > 0:
        print(f"  Deleting {len(results['ids'])} old {doc_type} docs for {ticker}")
        collection.delete(ids=results["ids"])


def ingestion_loop():
    """Run once to ingest/update data"""
    print("\n🔄 Refreshing stock knowledge base...\n")

    for ticker in TICKERS:
        try:
            print(f"Updating {ticker}")

            # Fundamentals
            delete_existing(ticker, "fundamentals")
            fundamentals = chunk_text(fetch_stock_data(ticker))
            index_documents(ticker, "fundamentals", fundamentals)

            # News
            delete_existing(ticker, "news")
            news_chunks = []
            for article in fetch_news_yahoo(ticker):
                news_chunks.extend(chunk_text(article))
            index_documents(ticker, "news", news_chunks)

        except Exception as e:
            print(f"❌ {ticker} failed: {e}")

    print(f"\n✅ Vector DB refreshed. Total docs: {len(collection.get()['ids'])}")


# ==========================
# RAG QUERY
# ==========================
def rag_query(question: str) -> str:
    print("QUERY is ", question)
    # 1️⃣ Compute embedding
    q_emb = ollama_embed([question])[0]

    # 2️⃣ Check if collection has any documents
    total_docs = len(collection.get()["ids"])
    if total_docs == 0:
        return "Vector DB is empty. Please run ingestion first."

    print(f"Querying against {total_docs} documents...")

    # 3️⃣ Query top-k nearest neighbors
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K
    )

    # 4️⃣ Build context
    context = "\n\n".join(
        f"[{m['ticker']} | {m['type']}]\n{doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    )

    # 5️⃣ Ask LLM
    prompt = f"""
You are a financial research assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    return ollama_generate(prompt)


# ==========================
# MAIN - IMPROVED
# ==========================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        # Run ingestion mode
        ingestion_loop()
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        # Run query mode
        print("\n📈 RAG system ready. Running test queries.\n")
        
        if len(TICKERS) >= 3:
            print("\n", rag_query(f"Tell me about {TICKERS[0]}"), "\n")
            print("="*50)
            print("\n", rag_query(f"What's the latest news on {TICKERS[1]}?"), "\n")
            print("="*50)
            print("\n", rag_query(f"Compare {TICKERS[0]} and {TICKERS[2]}"), "\n")
    else:
        print("""
Usage:
  python script.py ingest  - Ingest/update stock data
  python script.py query   - Run test queries
        """)