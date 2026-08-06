#!/usr/bin/env python3
"""
ingest.py

Reads a corpus of .txt/.md files, chunks them,
generates Gemini embeddings, and builds a local index.

Outputs:

index/
    vectors.npy
    metadata.json
    config.json
"""

import argparse
import json
from pathlib import Path
import faiss
import numpy as np

from chunker import chunk_document
from bedrock_llm import embed_text


def load_documents(corpus_dir: Path):
    """
    Load all .txt and .md files.
    """

    docs = []

    for ext in ("*.txt", "*.md"):
        for file in corpus_dir.glob(ext):

            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            docs.append(
                {
                    "doc_id": file.stem,
                    "path": str(file),
                    "text": text
                }
            )

    return docs

def build_index(
    corpus_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int
):
    """
    Build a FAISS IndexFlatIP index using Gemini embeddings.
    """

    print("=" * 60)
    print("Loading documents...")
    print("=" * 60)

    documents = load_documents(corpus_dir)

    print(f"Loaded {len(documents)} documents.\n")

    all_vectors = []
    metadata = []

    total_chunks = 0

    for doc in documents:

        print(f"Processing {doc['doc_id']}...")

        chunks = chunk_document(
            text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for idx, chunk in enumerate(chunks):

            print(
                f"  Embedding chunk {idx + 1}/{len(chunks)}",
                end="\r"
            )

            vector = embed_text(chunk)

            all_vectors.append(vector)

            metadata.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_id": idx,
                    "text": chunk
                }
            )

            total_chunks += 1

        print(f"  Finished ({len(chunks)} chunks)")

    print()

    print("=" * 60)
    print("Building FAISS Index...")
    print("=" * 60)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Convert to float32 NumPy array
    vectors = np.asarray(
        all_vectors,
        dtype=np.float32
    )

    # Normalize vectors so inner product == cosine similarity
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    # Exact cosine similarity index
    index = faiss.IndexFlatIP(dimension)

    print("Adding vectors...")
    index.add(vectors)

    print("Saving index...")

    faiss.write_index(
        index,
        str(output_dir / "index.faiss")
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Done.\n")

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {total_chunks}")
    print(f"Vectors   : {index.ntotal}")
    print(f"Dimension : {dimension}")
    print(f"Index Type: IndexFlatIP")
    print(f"Saved to  : {output_dir}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        required=True,
        help="Folder containing .txt/.md files"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output index directory"
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=1000,
        help="Chunk size"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Chunk overlap"
    )
    '''
    parser.add_argument(
        "--nlist",
        type=int,
    )
    parser.add_argument(
        "--pq_m",
        type=int,
    )
    parser.add_argument(
        "--pq_bits",
        type=int,
    )'''

    args = parser.parse_args()

    build_index(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.out),
        chunk_size=args.chunk,
        overlap=args.overlap
        #nlist=args.nlist,
        #pq_m=args.pq_m,
        #pq_bits=args.pq_bits
    )


if __name__ == "__main__":
    main()
