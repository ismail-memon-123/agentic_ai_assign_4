# Gemini RAG Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Google's Gemini models.

The pipeline:

1. Reads documents from a corpus directory
2. Splits documents into overlapping chunks
3. Generates embeddings using Gemini
4. Stores embeddings in a NumPy vector index
5. Embeds user queries
6. Performs cosine similarity search
7. Retrieves the top-k most relevant chunks
8. Builds a grounded prompt
9. Generates an answer using Gemini
10. Displays the answer with citations

---

# Project Structure

```
project/
│
├── corpus/
│   ├── doc1.txt
│   ├── doc2.md
│   └── ...
│
├── index/
│   ├── vectors.npy
│   ├── metadata.json
│   └── config.json
│
├── chunker.py
├── embeddings.py
├── ingest.py
├── index_utils.py
├── requirements.txt
└── README.md
```

---

# Requirements

- Python 3.10+
- Google Gemini API Key

Install dependencies:

```bash
pip install -r requirements.txt
```

Example requirements:

```
google-genai
numpy
tqdm
```

---

# Configure Gemini

Create an API key:

https://aistudio.google.com/app/apikey

Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Windows:

```cmd
set GEMINI_API_KEY=YOUR_API_KEY
```

---

# Creating the Index

Place documents inside

```
corpus/
```

Supported files:

- .txt
- .md

Run:

```bash
python ingest.py \
    --corpus ./corpus \
    --out ./index \
    --chunk 200 \
    --overlap 50
```

This generates

```
index/
    vectors.npy
    metadata.json
    config.json
```

---

# Querying

Ask a question:

```bash
python index_utils.py
When invoked ask the question
```

# How the Pipeline Works

```
Documents
      │
      ▼
Chunk Documents
      │
      ▼
Generate Gemini Embeddings
      │
      ▼
Save NumPy Index

===============================

User Question
      │
      ▼
Embed Query
      │
      ▼
Cosine Similarity
      │
      ▼
Top-K Chunks
      │
      ▼
Prompt Builder
      │
      ▼
Gemini
      │
      ▼
Final Answer
```

---

# Files

## chunker.py

Responsible for:

- reading documents
- cleaning text
- splitting into overlapping chunks

Returns

```
[
    chunk1,
    chunk2,
    ...
]
```

---

## bedrock_llm.py

Responsible for:

- connecting to Gemini
- generating embeddings
- invoking llm with prompts

## ingest.py

Creates the vector database.

Steps

1. Read documents
2. Chunk documents
3. Generate embeddings
4. Save vectors
5. Save metadata

Outputs

```
vectors.npy

metadata.json

config.json
```

---

## index_utils.py

Responsible for

- loading vectors
- loading metadata
- cosine similarity search
- returning top-k chunks
- creating full query
- asking user for question and answering it

---


# Example

Take a look at the screenshot, I made up a edm dj (slightly changed David Guetta's name and copied a part of an article about him
with the changed name. When asked about the fake DJ the information from the article is presented and the sources cited).

I also used chat gpt to generate a fake article about a fake country and asked questions about that.
# Troubleshooting

## GEMINI_API_KEY not found

Verify

```bash
echo $GEMINI_API_KEY
```

or

```cmd
echo %GEMINI_API_KEY%
```

---

## Empty Retrieval

Possible causes

- Chunk size too large
- Overlap too small
- Wrong embedding model
- Empty corpus

---
