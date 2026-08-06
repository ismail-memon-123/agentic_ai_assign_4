import numpy as np
import json
from dataclasses import asdict
from bedrock_llm import embed_text, generate_answer
from pydantic import BaseModel
import faiss
import time
import sys
class RAGResponse(BaseModel):

    answer: str
    citations: list[str]
    confidence: float
    safety_flags: list[str]

def normalize(vectors):
    """
    Normalize vectors for cosine similarity
        """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def initial(
    index_file="index/index.faiss",
    chunks_file="index/metadata.json"
):
    """
    Load the FAISS index and document metadata.
    """

    index = faiss.read_index(index_file)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {index.ntotal} vectors")
    print(f"Embedding dimension: {index.d}")

    return chunks, index

def search(index, chunks, query_embedding, top_k=5, nprobe=10):
    """
    Search the FAISS index and return the top-k most similar chunks.
    """

    # FAISS expects float32 with shape (1, dimension)
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Number of clusters to search
    #index.nprobe = nprobe

    # Search the index
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        # Ignore invalid entries
        if idx == -1:
            continue

        results.append(
            {
                "score": float(score),
                "text": chunks[idx]["text"],
                "id": chunks[idx]["chunk_id"],
                "doc_id": chunks[idx]["doc_id"]
            }
        )

    return results

def retrieve(question, top_k=5, nprobe=10):
    """
    Retrieve the most relevant document chunks for a question.
    """

    # Embed the user's question
    query_embedding = embed_text(question)

    # Load the FAISS index and metadata
    chunks, index = initial()

    # Search the index
    print("start search")
    start_time = time.perf_counter()
    results = search(
        index=index,
        chunks=chunks,
        query_embedding=query_embedding,
        top_k=top_k,
        nprobe=nprobe
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed search time: {elapsed_time:.6f} seconds")

    return results

def build_prompt(question, results):

    """
    Create grounded RAG prompt
    """

    context = ""

    for i, result in enumerate(results):

        citation = f"[Source {i+1}]"


        context += f"""
{citation}
{result['text']}

"""

    prompt = f"""

You are a helpful assistant.

Answer the user's question ONLY using the provided context.

If the answer is not contained in the context,
say "I don't have enough information."

Always cite sources using [Source X] format.

Return ONLY JSON.

{{
    "answer":"",
    "citations":[],
    "confidence":0,
    "safety_flags":[]
}}
Context:

{context}


Question:

{question}


Answer:

"""
    print(prompt)

    return prompt

def main(l=True, q=''):

    if l:
        question = input(
            "\nQuestion: "
        )
    else:
        question = q
    print(question)


    # 1. Retrieve documents

    results = retrieve(
        question
    )


    # 2. Build grounded prompt

    rag_prompt = build_prompt(
        question,
        results
    )


    # 3. Call LLM

    answer = generate_answer(rag_prompt)
    print("Not yet format validated")
    #answer = "fake ans"
    print(answer)
    try:
        answer = answer.replace("```json", "").replace("```", "").strip()
        result = RAGResponse.model_validate_json(answer)
    except Exception:

        # Retry once
        print("trying again")
        stricter_prompt = rag_prompt + """

            Return ONLY valid JSON.

        """

        answer = generate_answer(stricter_prompt)
        #answer = "faker"
        print(answer)

    try:
        answer = answer.replace("```json", "").replace("```", "").strip()
        result = RAGResponse.model_validate_json(answer)
    except:
        result = {
            "answer":"Unable to generate a valid response.",
            "citations":[],
            "confidence":0,
            "safety_flags":["validation_failed"]
        }

    # 4. Print answer

    print("\n====================")
    print("ANSWER")
    print("====================")

    print(result)



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(l=False, q=sys.argv[1])
    else:
        main()
