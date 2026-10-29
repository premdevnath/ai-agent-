"""
Tests for the sentence-aware token-budget chunker.

Run with pytest, or standalone (no test framework needed):
    python tests/test_chunking.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunking import (
    DEFAULT_CHUNK_TOKENS,
    chunk_documents,
    chunk_text,
    count_tokens,
    split_sentences,
)

PROSE = (
    "Acme Corp reported total revenue of $4.2 billion in Q3 FY2025, up 18% "
    "year-over-year. Gross margin expanded to 62.4%, compared to 59.1% in "
    'the prior-year quarter. "We are prioritizing agentic AI tooling," the '
    "CEO said. Capital expenditures totaled $612 million, largely allocated "
    "to GPU capacity.\n\nKey risk factors include foreign currency exposure "
    "and dependence on a limited number of hyperscale cloud partners. "
    "Management guided full-year revenue to $16.8-$17.2 billion."
)

# PDF-style extraction: hard line wraps mid-sentence, which must NOT become
# sentence boundaries.
HARD_WRAPPED = (
    "Total long-term debt was $5.4 billion at\n"
    "quarter end, with a weighted average interest\n"
    "rate of 4.1% and no material maturities due\n"
    "before FY2027. The Board authorized an\n"
    "additional $3.0 billion share repurchase program."
)

UNICODE_TEXT = (
    "Revenue in the EMEA région grew 12% — driven by München and São Paulo "
    "offices. 営業利益は前年比15%増加した。 Margins improved across the board."
)

DOCUMENT_BATTERY = [
    PROSE,
    HARD_WRAPPED,
    UNICODE_TEXT,
    "One sentence only.",
    "no terminal punctuation at all just a heading",
    "word " * 3000,  # far over any budget, no sentence boundaries
    "x" * 5000,  # single unbroken blob
    "",
    "   \n\n  \n ",
    "A. B. C. D. E. F. G. H.",  # many tiny sentences
]


def test_split_sentences_basic():
    segments = split_sentences(PROSE)
    assert len(segments) == 6
    assert segments[0].startswith("Acme Corp")
    assert segments[0].endswith("year-over-year.")
    # Quoted sentence end is one boundary, not two.
    assert any(s.startswith('"We are prioritizing') for s in segments)


def test_split_sentences_ignores_hard_line_wraps():
    segments = split_sentences(HARD_WRAPPED)
    assert len(segments) == 2
    assert segments[0].startswith("Total long-term debt")
    assert segments[0].endswith("before FY2027.")


def test_split_sentences_paragraph_breaks():
    segments = split_sentences("First heading\n\nSecond paragraph starts here")
    assert segments == ["First heading", "Second paragraph starts here"]


def test_budget_is_hard_guarantee_across_battery():
    for budget in (32, 64, 256):
        for doc in DOCUMENT_BATTERY:
            for chunk in chunk_text(doc, max_tokens=budget):
                assert count_tokens(chunk) <= budget, (
                    f"budget {budget} exceeded: {count_tokens(chunk)} tokens"
                )


def test_no_text_dropped():
    for doc in DOCUMENT_BATTERY:
        chunks = chunk_text(doc, max_tokens=64)
        joined = " ".join(chunks)
        # Every word of the input survives into some chunk, in order.
        # (Whitespace is normalized at segment edges, so compare word lists;
        # the token-window fallback may split inside a "word", so compare
        # the concatenated non-whitespace characters instead.)
        assert "".join(joined.split()) == "".join(doc.split())


def test_sentences_never_split_when_they_fit():
    chunks = chunk_text(PROSE, max_tokens=48)
    sentence_ends = [s[-1] for s in split_sentences(PROSE)]
    for chunk in chunks:
        # Each chunk ends exactly where one of the sentences ends.
        assert chunk[-1] in sentence_ends


def test_packing_is_greedy():
    # With a huge budget everything packs into one chunk.
    chunks = chunk_text(PROSE, max_tokens=100_000)
    assert len(chunks) == 1


def test_overlap_restarts_previous_sentences():
    chunks = chunk_text(PROSE, max_tokens=48, overlap_sentences=1)
    assert len(chunks) >= 2
    overlapped = 0
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_sentences = split_sentences(prev)
        if len(prev_sentences) < 2:
            # A single-sentence chunk cannot be overlapped: the rewind is
            # capped so packing always advances (no infinite loop).
            continue
        assert nxt.startswith(prev_sentences[-1][:20])
        overlapped += 1
    assert overlapped >= 1


def test_overlap_always_advances():
    # Overlap larger than the sentences per chunk must still terminate
    # and cover the document.
    chunks = chunk_text(PROSE, max_tokens=32, overlap_sentences=50)
    assert len(chunks) < 100
    last_words = PROSE.split()[-2:]
    assert " ".join(last_words) in chunks[-1]


def test_oversized_sentence_falls_back_to_token_windows():
    blob = "word " * 3000
    chunks = chunk_text(blob, max_tokens=64)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 64 for c in chunks)


def test_empty_and_whitespace_inputs():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_invalid_arguments_rejected():
    for call in (
        lambda: chunk_text(PROSE, max_tokens=0),
        lambda: chunk_text(PROSE, overlap_sentences=-1),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_determinism():
    for doc in DOCUMENT_BATTERY:
        assert chunk_text(doc, max_tokens=64) == chunk_text(doc, max_tokens=64)


def test_chunk_documents_preserves_page_metadata():
    from langchain_core.documents import Document

    pages = [
        Document(page_content=PROSE, metadata={"source_file": "10k.pdf", "page": 1}),
        Document(page_content=HARD_WRAPPED, metadata={"source_file": "10k.pdf", "page": 2}),
    ]
    chunks = chunk_documents(pages, max_tokens=48)
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "10k.pdf"
        assert chunk.metadata["page"] in (1, 2)
    # Chunks from page 2 carry page 2's metadata, not page 1's.
    page2_chunks = [c for c in chunks if c.metadata["page"] == 2]
    assert any("long-term debt" in c.page_content for c in page2_chunks)
    # Pages are chunked independently: no chunk mixes text from both pages.
    for chunk in chunks:
        assert not ("Acme Corp" in chunk.page_content
                    and "long-term debt" in chunk.page_content)


def test_default_budget_matches_app_config():
    chunks = chunk_text(PROSE * 20)
    assert all(count_tokens(c) <= DEFAULT_CHUNK_TOKENS for c in chunks)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
