# Gemini RAG Pipeline

## Overview

This project implements a production-style Retrieval-Augmented Generation (RAG) system using Google's Gemini API and FAISS.

The application ingests a collection of documents, generates semantic embeddings using Gemini, stores them in a FAISS vector database, retrieves the most relevant document chunks for a user query, and generates grounded responses using Gemini.

The project also implements several production-oriented features including:

- FAISS vector indexing
- Gemini embeddings
- Retrieval-Augmented Generation (RAG)
- Gemini Safety Guardrails
- Structured JSON output validation
- Timestamped logging
- Source citations

---

# Assignment Features

This implementation satisfies the following production RAG requirements.

## Retrieval

- Gemini embedding model
- FAISS IndexFlatIP vector database
- Cosine similarity search
- Top-K document retrieval

## Generation

- Gemini Large Language Model
- Grounded prompt construction
- Source citations
- Structured JSON responses

## Safety

- Gemini Safety Settings
- Harmful content filtering
- Safety ratings returned from Gemini

## Validation

All responses are validated using Pydantic before being returned.

## Logging

Timestamped logs are generated for

- Embedding requests
- Gemini safety information
- API failures

---

# System Architecture

```
                    DOCUMENT INGESTION

        Documents
             │
             ▼
      Document Loader
             │
             ▼
      Document Chunker
             │
             ▼
     Gemini Embeddings
             │
             ▼
     FAISS IndexFlatIP
             │
             ▼
      Save index.faiss

====================================================

                    QUERY PIPELINE

        User Question
             │
             ▼
       Gemini Embedding
                    │
                    ▼
              FAISS Search
                    │
                    ▼
             Top-K Chunks
                    │
                    ▼
            Prompt Builder
                    │
                    ▼
        Gemini Safety Settings
                    │
                    ▼
          Gemini Generation
                    │
                    ▼
         Pydantic Validation
                    │
                    ▼
           Cache Response
                    │
                    ▼
             Final Response
```

---

# Project Structure

```
project/

│
├── corpus/
│     ├── document1.txt
│     ├── document2.md
│
├── index/
│     ├── index.faiss
│     └── metadata.json
│
│
├── logs/
│     ├── gemini.log
│     └── prompt_cache.log
│
├── chunker.py
├── bedrock_llm.py
├── ingest.py
├── index_utils.py
├── requirements.txt
└── README.md
---

# Requirements

- Python 3.10+
- Google Gemini API Key

Install dependencies

```bash
pip install -r requirements.txt
```

Example requirements

```
google-genai
faiss-cpu
numpy
pydantic
tqdm
```

---

# Configure Gemini

Create an API Key

https://aistudio.google.com/app/apikey

Linux/macOS

```bash
export GEMINI_API_KEY=YOUR_API_KEY
```

Windows

```cmd
set GEMINI_API_KEY=YOUR_API_KEY
```

---

# Supported Documents

The ingestion pipeline currently supports

- TXT
- Markdown

Simply place the documents inside

```
corpus/
```

---

# Building the Index

Generate embeddings and build the FAISS index.

```bash
python ingest.py \
    --corpus ./corpus \
    --out ./index \
    --chunk 200 \
    --overlap 50
```

The ingestion process

1. Reads every document
2. Splits the document into overlapping chunks
3. Generates Gemini embeddings
4. Normalizes embeddings
5. Builds a FAISS IndexFlatIP index
6. Stores document metadata

Output

```
index/

    index.faiss

    metadata.json
```

---

# Querying

Run

```bash
python index_utils.py
```

The application prompts for a question or you can enter question as second argument. Other option is using the run_questions.sh
shell script and provide a text file of questions (one on each line) as input, and it will go through all the questions one by one.

```
Question:

Who founded Blue Horizon Analytics?
```

The pipeline then

1. Generates a Gemini embedding
3. Searches FAISS
4. Retrieves the Top-K chunks
5. Builds a grounded prompt
6. Sends the prompt through Gemini Safety
7. Generates an answer
8. Validates the JSON response
9. Returns the answer

---

# Gemini Safety Guardrails

All prompts are submitted using Gemini Safety Settings.

The following safety categories are enabled

- Harassment
- Hate Speech
- Dangerous Content
- Sexually Explicit Content

Safety ratings returned by Gemini are recorded in the application logs.

---

# JSON Validation

Every Gemini response is validated using Pydantic.

Expected response format

```json
{
    "answer": "...",
    "citations": [
        "Source 1"
    ],
    "confidence": 1,
    "safety_flags": []
}
```

Responses that do not satisfy the schema are rejected before being returned.

---

# Logging

The application maintains timestamped log files.

## Gemini Log

```
logs/

    gemini.log
```

Records

- Embedding requests
- Generation requests
- Safety ratings
- API failures

Example

```
[2026-08-05 19:42:10] Embedding Request
[2026-08-05 19:42:11] Embedding Success
[2026-08-05 19:42:12] Generation Request
[2026-08-05 19:42:13] Generation Success
```


---

# Project Files

## chunker.py

Responsible for

- Reading documents
- Cleaning text
- Splitting documents into overlapping chunks

---

## bedrock_llm.py

Responsible for

- Connecting to Gemini
- Generating embeddings
- Invoking the Gemini LLM
- Applying Gemini Safety Settings
- Logging embedding and generation requests

---

## ingest.py

Responsible for

- Reading documents
- Chunking documents
- Generating embeddings
- Building the FAISS index
- Saving metadata

Output

```
index.faiss

metadata.json
```

---

## index_utils.py

Responsible for

- Loading the FAISS index
- Loading metadata
- Embedding user questions
- Performing Top-K similarity search
- Building the grounded prompt
- Returning retrieved chunks
- Deals with building prompt
- Answers user
---


# Example Workflow

1. User asks a question.
2. The question is embedded using Gemini.
5. FAISS retrieves the Top-K matching chunks.
6. A grounded prompt is created.
7. Gemini generates a response using Safety Settings.
8. The response is validated using Pydantic.
9. The final answer and citations are displayed.

---

# Troubleshooting

## GEMINI_API_KEY not found

Verify

Linux

```bash
echo $GEMINI_API_KEY
```

Windows

```cmd
echo %GEMINI_API_KEY%
```

---

## Empty Retrieval

Possible causes

- Empty corpus
- Incorrect chunk size
- Incorrect overlap

---

## FAISS Import Error

Install FAISS

```bash
pip install faiss-cpu
```

If using Windows, ensure that your Python version is supported by the installed FAISS package.

## Report
A detailed report is included in the report file. It tests questions.txt and questions_negative.txt
