"""
Sentence-Aware Token-Budget Chunker
====================================
Splits page text into chunks of whole sentences packed up to a hard token
budget, measured with the same tokenizer the embedding model uses
(cl100k_base for text-embedding-3-small).

Why sentence packing instead of a character splitter: in a controlled,
budget-matched benchmark of chunking strategies (rag-chunking-bench in
https://github.com/kantamaniprakash/genai-lab), sentence-packed 256-token
chunks were the strongest configuration on long-reference corpora —
including a financial-documents corpus — at the retrieved-token budgets
this app runs at. Full rationale and numbers are in the README's
"Chunking configuration" section.

Guarantees:
- No chunk ever exceeds the token budget (a single sentence longer than
  the budget is split by a token window as a fallback).
- No text is dropped: every non-whitespace segment of the input lands in
  exactly one chunk (or, with overlap enabled, at least one).
- Deterministic: same input, same chunks.

Author: Satya Sai Prakash Kantamani
GitHub: https://github.com/kantamaniprakash
"""

import re
from functools import lru_cache

import tiktoken

DEFAULT_CHUNK_TOKENS = 256
DEFAULT_OVERLAP_SENTENCES = 0
ENCODING_NAME = "cl100k_base"  # tokenizer of text-embedding-3-small / gpt-4o-mini

# A sentence boundary is terminal punctuation (optionally followed by closing
# quotes/brackets) then whitespace then an uppercase/digit/quote start, or a
# blank line (paragraph break). Single newlines are NOT boundaries: PDF text
# extraction hard-wraps lines mid-sentence, and splitting there would put
# chunk boundaries inside sentences — the exact failure this chunker exists
# to avoid. Known failure modes (abbreviations like "Dr.", initials) only
# over-split, and over-split segments get re-packed into the same chunk.
_SENT_BOUNDARY_RE = re.compile(r'[.!?]["\')\]]*\s+(?=[A-Z0-9"\'(\[])|\n\s*\n')


@lru_cache(maxsize=1)
def _encoding() -> "tiktoken.Encoding":
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Token count under the embedding model's tokenizer.

    Special-token strings occurring in document text (e.g. "<|endoftext|>"
    inside a scraped PDF) are encoded as ordinary text, never as controls.
    """
    return len(_encoding().encode(text, disallowed_special=()))


def split_sentences(text: str) -> list:
    """Split text into whitespace-stripped sentence segments, in order."""
    boundaries = [0]
    for m in _SENT_BOUNDARY_RE.finditer(text):
        boundaries.append(m.end())
    boundaries.append(len(text))
    segments = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        stripped = text[lo:hi].strip()
        if stripped:
            segments.append(stripped)
    return segments


def _split_oversized(segment: str, max_tokens: int) -> list:
    """Token-window fallback for a single segment over the budget.

    Decoding a token slice and re-encoding it does not always round-trip to
    the same count, so each window is re-checked and recursively re-split in
    the (rare) case it still exceeds the budget.
    """
    enc = _encoding()
    tokens = enc.encode(segment, disallowed_special=())
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        piece = enc.decode(tokens[start:start + max_tokens]).strip()
        if not piece:
            continue
        if count_tokens(piece) > max_tokens:
            pieces.extend(_split_oversized(piece, max(1, max_tokens // 2)))
        else:
            pieces.append(piece)
    return pieces


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list:
    """Pack whole sentences greedily into chunks of at most max_tokens.

    The budget is checked on the actual joined chunk text (not a sum of
    per-sentence counts) so BPE merges across joins can never push a chunk
    over. `overlap_sentences` restarts each chunk that many sentences before
    the previous chunk ended.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    pieces = []
    for segment in split_sentences(text):
        if count_tokens(segment) > max_tokens:
            pieces.extend(_split_oversized(segment, max_tokens))
        else:
            pieces.append(segment)
    if not pieces:
        return []

    chunks = []
    i = 0
    while i < len(pieces):
        current = pieces[i]
        j = i + 1
        while j < len(pieces):
            candidate = current + " " + pieces[j]
            if count_tokens(candidate) > max_tokens:
                break
            current = candidate
            j += 1
        chunks.append(current)
        # Overlap rewinds the start, but never so far that we stop advancing.
        i = max(j - overlap_sentences, i + 1)
    return chunks


def chunk_documents(
    pages: list,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list:
    """Chunk a list of per-page LangChain Documents, preserving metadata.

    Chunking is per page so every chunk keeps an exact (source_file, page)
    citation; the cost is that a sentence straddling a page break is split
    at the break.
    """
    from langchain_core.documents import Document

    chunks = []
    for page in pages:
        for text in chunk_text(page.page_content, max_tokens, overlap_sentences):
            chunks.append(Document(page_content=text, metadata=dict(page.metadata)))
    return chunks
