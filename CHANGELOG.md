# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Replaced the character-based text splitter (RecursiveCharacterTextSplitter, 1000 chars / 200 overlap) with a sentence-aware chunker (`chunking.py`) that packs whole sentences into chunks of at most 256 tokens, counted with the embedding model's own tokenizer (cl100k_base). Defaults come from budget-matched retrieval benchmarks in [rag-chunking-bench](https://github.com/kantamaniprakash/genai-lab/tree/main/rag-chunking-bench); rationale and numbers are in the README's "Chunking configuration" section.

### Added
- Invariant tests for the chunker (`tests/test_chunking.py`): hard token budget, no dropped text, sentence-boundary alignment, overlap semantics, page-metadata preservation, determinism. Runnable standalone or under pytest, no API key required.

## [0.1.0] - 2026-07-05

### Added
- Retrieval-Augmented Generation chatbot for financial documents, built on LangChain, OpenAI GPT-4o-mini, ChromaDB, and a dark-themed Streamlit UI.
- Multi-PDF ingestion via PyMuPDF (read from bytes for cross-platform compatibility), with RecursiveCharacterTextSplitter chunking (1000 chars, 200 overlap).
- Semantic retrieval over a persistent ChromaDB store using MMR (top-5 from a candidate pool of 15) and `text-embedding-3-small` embeddings.
- Conversational memory with a sliding window of the last 5 turns for multi-turn follow-up questions.
- Source-cited answers showing the originating file name and page number for every response.
- Standalone evaluation harness (`eval_harness.py`) that scores retrieval (Hit Rate@k, MRR@k) and generation (faithfulness, answer relevancy, keyword coverage) against a built-in synthetic corpus and labeled eval set.
- GitHub Actions CI (lint + tests on Python 3.10–3.12), Dependabot updates, MIT license, and smoke tests.

[0.1.0]: https://github.com/Kantamaniprakash/financial-rag-chatbot/releases/tag/v0.1.0

## Local Windows usability update

- Added `RUN_WINDOWS.bat` to create a virtual environment, install Python dependencies, and launch Streamlit.
- Added `START_HERE_HINGLISH.md` with beginner-friendly Windows setup and troubleshooting.
- Improved PDF validation and clearer messages for password-protected, invalid, empty, and scanned PDFs.
- Prevented duplicate chunk accumulation by rebuilding the named Chroma collection when processing documents again.
- Updated app footer to show Prem Devnath as the customizer while retaining original-author attribution.
