#!/usr/bin/env python3
"""
chunker.py

Utility functions for splitting documents into overlapping chunks
for Retrieval-Augmented Generation (RAG).

Example:
    from chunker import chunk_document

    chunks = chunk_document(
        text,
        chunk_size=1000,
        overlap=150
    )
"""

from typing import List


def clean_text(text: str) -> str:
    """
    Normalize whitespace in the document.
    """

    if not text:
        return ""

    # Remove carriage returns
    text = text.replace("\r", "")

    # Collapse multiple blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # Replace tabs
    text = text.replace("\t", " ")

    # Collapse repeated spaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def chunk_document(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[str]:
    """
    Split a document into overlapping chunks.

    Parameters
    ----------
    text : str
        Document text.

    chunk_size : int
        Maximum characters per chunk.

    overlap : int
        Number of characters to overlap.

    Returns
    -------
    List[str]
    """

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try not to split words unless we're at the end
        if end < text_length:

            space = text.rfind(" ", start, end)

            if space > start:
                end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Finished
        if end >= text_length:
            break

        # Move back for overlap
        start = max(0, end - overlap)

    return chunks


def chunk_file(
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[str]:
    """
    Read a text file and return chunks.
    """

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return chunk_document(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap
    )


def print_chunk_summary(chunks: List[str]) -> None:
    """
    Print chunk statistics.
    """

    print("=" * 50)
    print(f"Total Chunks : {len(chunks)}")
    print("=" * 50)

    for i, chunk in enumerate(chunks):

        print(
            f"Chunk {i:03d} "
            f"({len(chunk)} chars)"
        )


if __name__ == "__main__":

    sample = """
    Retrieval-Augmented Generation (RAG) combines
    document retrieval with language models.

    Documents are first split into chunks.

    Each chunk is converted into an embedding.

    During querying, the most relevant chunks
    are retrieved and passed to the language model.
    """ * 20

    chunks = chunk_document(
        sample,
        chunk_size=300,
        overlap=50
    )

    print_chunk_summary(chunks)

    print("\nFirst Chunk\n")
    print(chunks[0])
