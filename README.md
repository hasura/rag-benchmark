# RAG benchmark

## Generating data

See the script `load.py` which scrapes wiki articles and stores it in Postgres. It uses the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) dataset available in `test.tsv`

## Comparing

1. Naive RAG: `rag.py` is implementation of a naive RAG technique on top-k chunks of wikipedia articles
2. Naive RAG on titles: `rag_titles.py` is implementation of a RAG technique which gets top-k complete articles
3. Agentic RAG: `agentic_rag.py` is implemnetation of an agentic RAG technique on top-k chunks of wikipedia articles
4. Agentic RAG on titles: `agentic_rag_titles.py` is implemetation of an agentic RAG technique on top-k complete articles

## PromptQL setup

1. See the PromptQL project in `my-assistant/` directory
2. Run it via `ddn run docker-start`
