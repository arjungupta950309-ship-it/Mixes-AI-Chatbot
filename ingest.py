"""Build the Chroma vector store from scraped product data.

Each product becomes one rich, self-contained document (title + price +
rating + dietary badges + description). Products are short, so one document
per product is the right granularity — it keeps each retrieved chunk fully
coherent and avoids splitting a description mid-sentence.

Run once after scraping:  python ingest.py
"""

from __future__ import annotations

import json
import shutil

from langchain_core.documents import Document
from langchain_chroma import Chroma

import config
from providers import get_embeddings


def product_to_text(p: dict) -> str:
    """Render a product as a readable, retrieval-friendly text block."""
    lines = [f"Product: {p['title']}", f"Category: {p.get('category', 'Mixes')}"]
    if p.get("price"):
        lines.append(f"Price: {p['price']}")
    if p.get("rating"):
        reviews = p.get("review_count") or 0
        lines.append(f"Rating: {p['rating']} out of 5 ({reviews} reviews)")
    if p.get("badges"):
        pretty = ", ".join(BADGE_LABELS.get(b, b) for b in p["badges"])
        lines.append(f"Attributes: {pretty}")
    if p.get("description"):
        lines.append(f"Description: {p['description']}")
    lines.append(f"URL: {p['url']}")
    return "\n".join(lines)


# Human-readable labels for the raw badge slugs.
BADGE_LABELS = {
    "glutenfree": "Gluten-Free",
    "kosherpareve": "Kosher (Pareve)",
    "kosherdairy": "Kosher (Dairy)",
    "nongmo": "Non-GMO",
    "sourcednongmo": "Sourced Non-GMO",
    "organic": "Organic",
    "wholegrain": "Whole Grain",
    "wholegrain50": "50%+ Whole Grain",
    "wholegrain100": "100% Whole Grain",
    "madeintheusa": "Made in the USA",
    "vegan": "Vegan",
}


def build_documents() -> list[Document]:
    products = json.loads(config.DATA_PATH.read_text(encoding="utf-8"))
    docs: list[Document] = []
    for p in products:
        metadata = {
            "title": p["title"],
            "url": p["url"],
            "price": p.get("price", ""),
            "rating": p.get("rating") or 0,
            "review_count": p.get("review_count") or 0,
            "badges": ", ".join(p.get("badges", [])),
            "image": p.get("image", ""),
        }
        docs.append(Document(page_content=product_to_text(p), metadata=metadata))
    return docs


def main() -> None:
    docs = build_documents()
    print(f"Loaded {len(docs)} product documents from {config.DATA_PATH.name}")

    # Rebuild cleanly so re-runs are deterministic.
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    embeddings = get_embeddings()
    print(f"Embedding via '{config.EMBED_PROVIDER}' and writing to {config.CHROMA_DIR.name}/ ...")

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )

    print(f"Done. Vector store with {len(docs)} documents persisted to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
