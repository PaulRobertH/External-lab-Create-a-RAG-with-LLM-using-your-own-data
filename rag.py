#!/usr/bin/env python3
"""
Applied RAG over my own wine portfolio (Nexus stock export).

Pipeline: Pandas -> list of dicts -> Sentence Transformers embeddings
          -> Qdrant in-memory retrieval -> LLM generation (Llamafile / OpenAI).

Usage:
    python rag.py                                  # interactive
    python rag.py "which Italian reds are ready?"  # one-off question

Start a local OpenAI-compatible LLM first, e.g. Llamafile:
    ./phi-2.Q4_K_M.llamafile --server --nobrowser   # serves http://localhost:8080/v1
"""

import json
import sys

import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/wine_portfolio.csv"
COLLECTION = "wine_portfolio"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_BASE_URL = "http://localhost:8080/v1"  # Llamafile / OpenAI-compatible server

SYSTEM_PROMPT = (
    "You are a knowledgeable wine cellar assistant for a private portfolio. "
    "Answer the user's question using ONLY the wine context provided. "
    "Refer to specific wines by name and vintage and briefly justify each pick. "
    "If the context has no good match, say so honestly."
)


def load_data(path=DATA_PATH):
    """Load the CSV with Pandas and return it as a list of dictionaries."""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def build_index(data, encoder):
    """Embed every description and store vectors + payloads in in-memory Qdrant."""
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )
    client.upload_points(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=idx,
                vector=encoder.encode(doc["description"]).tolist(),
                payload=doc,
            )
            for idx, doc in enumerate(data)
        ],
    )
    return client


def search(client, encoder, query, limit=4):
    """Retrieve the most semantically similar wines for a query."""
    hits = client.query_points(
        collection_name=COLLECTION,
        query=encoder.encode(query).tolist(),
        limit=limit,
    ).points
    return [hit.payload for hit in hits]


def generate(llm, question, context):
    """Augment the prompt with retrieved context and ask the LLM to answer."""
    user_message = (
        f"Question: {question}\n\n"
        f"Wine context (JSON):\n{json.dumps(context, indent=2)}"
    )
    completion = llm.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content


def main():
    print("Loading portfolio and building the vector index...")
    data = load_data()
    encoder = SentenceTransformer(EMBED_MODEL)
    client = build_index(data, encoder)
    llm = OpenAI(base_url=LLM_BASE_URL, api_key="sk-no-key-required")
    print(f"Indexed {len(data)} wines. Ask away (Ctrl-C to quit).\n")

    if len(sys.argv) > 1:
        _answer(client, encoder, llm, " ".join(sys.argv[1:]))
        return

    try:
        while True:
            question = input("You: ").strip()
            if question:
                _answer(client, encoder, llm, question)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


def _answer(client, encoder, llm, question):
    context = search(client, encoder, question)
    try:
        answer = generate(llm, question, context)
    except Exception as exc:  # noqa: BLE001
        print(
            f"\n[Could not reach the LLM at {LLM_BASE_URL}: {exc}]\n"
            "Start Llamafile (or set LLM_BASE_URL to a valid OpenAI endpoint) and try again.\n"
            "Retrieved wines were: "
            + ", ".join(f"{p['name']} {p['vintage']}" for p in context)
            + "\n"
        )
        return
    print(f"\nAssistant: {answer}")
    print("Sources: " + ", ".join(f"{p['name']} {p['vintage']}" for p in context) + "\n")


if __name__ == "__main__":
    main()
