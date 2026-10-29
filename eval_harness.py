"""
RAG Evaluation Harness
=======================
Component-level evaluation for the Financial RAG Chatbot's retrieval and
generation stages, evaluated separately so failures can be attributed to
the right part of the pipeline.

Retrieval metrics (does the retriever surface the right chunks?):
  - Hit Rate@k  — fraction of questions where a relevant chunk appears in the top-k
  - MRR@k       — mean reciprocal rank of the first relevant chunk

Generation metrics (given good context, does the model answer well?):
  - Faithfulness   — LLM-judge score (0-1) for whether the answer is grounded
                      in the retrieved context, with no unsupported claims
  - Answer Relevancy — LLM-judge score (1-5) for how directly the answer
                      addresses the question

The harness ships with its own fixture corpus and labeled eval set, so it
runs standalone against a fresh in-memory vector store — no uploaded PDFs
required.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval_harness.py
    python eval_harness.py --k 3 --output results.json
"""

import os
import sys
import json
import argparse
from statistics import mean

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ─── Fixture Corpus ─────────────────────────────────────────────────────────
# Synthetic financial-report passages, each tagged with a stable chunk_id so
# retrieval hits can be checked deterministically.
CORPUS = [
    {"chunk_id": "revenue_q3", "source_file": "acme_10k_2025.pdf", "page": 12,
     "text": "Acme Corp reported total revenue of $4.2 billion in Q3 FY2025, "
             "up 18% year-over-year, driven primarily by cloud services "
             "growth of 31% and steady demand in the enterprise segment."},
    {"chunk_id": "gross_margin", "source_file": "acme_10k_2025.pdf", "page": 14,
     "text": "Gross margin expanded to 62.4% in Q3 FY2025, compared to 59.1% "
             "in the prior-year quarter, reflecting a favorable shift toward "
             "higher-margin subscription revenue and lower component costs."},
    {"chunk_id": "risk_factors", "source_file": "acme_10k_2025.pdf", "page": 27,
     "text": "Key risk factors disclosed include exposure to foreign currency "
             "fluctuations, dependence on a limited number of hyperscale "
             "cloud partners, and potential disruption from new AI-related "
             "export control regulations."},
    {"chunk_id": "capex", "source_file": "acme_10k_2025.pdf", "page": 33,
     "text": "Capital expenditures totaled $612 million in Q3 FY2025, largely "
             "allocated to expanding GPU capacity in the company's three "
             "primary data center regions."},
    {"chunk_id": "guidance", "source_file": "acme_10k_2025.pdf", "page": 41,
     "text": "Management guided full-year FY2025 revenue to a range of "
             "$16.8 billion to $17.2 billion, implying full-year growth of "
             "approximately 20% at the midpoint."},
    {"chunk_id": "ai_strategy", "source_file": "acme_earnings_call.pdf", "page": 3,
     "text": "On the earnings call, the CEO said the company is prioritizing "
             "investment in agentic AI tooling and expects internal "
             "productivity gains of 15-20% from AI-assisted engineering "
             "workflows by the end of FY2026."},
    {"chunk_id": "segment_comparison", "source_file": "acme_earnings_call.pdf", "page": 6,
     "text": "The CFO noted that the Cloud segment now carries a 68% gross "
             "margin versus 41% for the legacy Hardware segment, and that "
             "the mix shift toward Cloud is the primary driver of "
             "consolidated margin expansion."},
    {"chunk_id": "headcount", "source_file": "acme_earnings_call.pdf", "page": 9,
     "text": "Total headcount stood at 48,200 employees at quarter end, down "
             "3% year-over-year following a restructuring in the Hardware "
             "division announced in Q1 FY2025."},
    {"chunk_id": "buyback", "source_file": "acme_10k_2025.pdf", "page": 45,
     "text": "The Board authorized an additional $3.0 billion share "
             "repurchase program, supplementing $1.1 billion remaining "
             "under the prior authorization."},
    {"chunk_id": "debt", "source_file": "acme_10k_2025.pdf", "page": 52,
     "text": "Total long-term debt was $5.4 billion at quarter end, with a "
             "weighted average interest rate of 4.1% and no material "
             "maturities due before FY2027."},
]

# ─── Labeled Eval Set ───────────────────────────────────────────────────────
# Each item has the ground-truth chunk(s) that should be retrieved and a
# short list of facts a correct answer must contain.
EVAL_SET = [
    {"question": "What was Acme's revenue growth in Q3 FY2025?",
     "relevant_chunk_ids": ["revenue_q3"],
     "expected_facts": ["$4.2 billion", "18%"]},
    {"question": "How did gross margin change year-over-year?",
     "relevant_chunk_ids": ["gross_margin"],
     "expected_facts": ["62.4%", "59.1%"]},
    {"question": "What are the key risk factors mentioned in the 10-K?",
     "relevant_chunk_ids": ["risk_factors"],
     "expected_facts": ["foreign currency", "cloud partners"]},
    {"question": "How much did the company spend on capital expenditures?",
     "relevant_chunk_ids": ["capex"],
     "expected_facts": ["$612 million", "GPU capacity"]},
    {"question": "What is management's full-year revenue guidance?",
     "relevant_chunk_ids": ["guidance"],
     "expected_facts": ["$16.8 billion", "$17.2 billion"]},
    {"question": "What did the CEO say about AI investment on the earnings call?",
     "relevant_chunk_ids": ["ai_strategy"],
     "expected_facts": ["agentic AI", "15-20%"]},
    {"question": "How do the Cloud and Hardware segment margins compare?",
     "relevant_chunk_ids": ["segment_comparison"],
     "expected_facts": ["68%", "41%"]},
    {"question": "What is the company's current headcount?",
     "relevant_chunk_ids": ["headcount"],
     "expected_facts": ["48,200"]},
    {"question": "What is the size of the new share buyback authorization?",
     "relevant_chunk_ids": ["buyback"],
     "expected_facts": ["$3.0 billion"]},
    {"question": "What is the company's total long-term debt and average interest rate?",
     "relevant_chunk_ids": ["debt"],
     "expected_facts": ["$5.4 billion", "4.1%"]},
]

GENERATION_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant. \
Answer the question using ONLY the provided context. Be precise and cite \
specific numbers. If the context is insufficient, say so.

Context:
{context}

Question: {question}

Answer:"""

FAITHFULNESS_JUDGE_PROMPT = """You are grading whether an AI-generated answer is \
faithful to its source context (i.e. contains no claims that are not \
supported by the context).

Context:
{context}

Question: {question}

Answer to grade:
{answer}

Respond with ONLY a JSON object of the form:
{{"faithful": true/false, "reasoning": "<one sentence>"}}"""

RELEVANCY_JUDGE_PROMPT = """You are grading how directly an AI-generated answer \
addresses the question asked, on a 1-5 scale (5 = fully and directly \
answers the question, 1 = does not address the question at all).

Question: {question}

Answer to grade:
{answer}

Respond with ONLY a JSON object of the form:
{{"score": <1-5 integer>, "reasoning": "<one sentence>"}}"""


def build_vector_store(api_key: str) -> Chroma:
    """Embed the fixture corpus into an ephemeral (non-persistent) Chroma store."""
    docs = [
        Document(
            page_content=item["text"],
            metadata={
                "chunk_id": item["chunk_id"],
                "source_file": item["source_file"],
                "page": item["page"],
            },
        )
        for item in CORPUS
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    return Chroma.from_documents(documents=docs, embedding=embeddings)


def retrieve_chunk_ids(vector_store: Chroma, question: str, k: int) -> list:
    docs = vector_store.similarity_search(question, k=k)
    return [doc.metadata["chunk_id"] for doc in docs], docs


def hit_rate(retrieved_ids: list, relevant_ids: list) -> bool:
    return any(rid in retrieved_ids for rid in relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def generate_answer(llm: ChatOpenAI, question: str, context_docs: list) -> str:
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = GENERATION_SYSTEM_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).content.strip()


def _parse_json_response(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge response: {raw!r}")
    return json.loads(raw[start:end + 1])


def judge_faithfulness(llm: ChatOpenAI, question: str, context_docs: list, answer: str) -> dict:
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    return _parse_json_response(llm.invoke(prompt).content)


def judge_relevancy(llm: ChatOpenAI, question: str, answer: str) -> dict:
    prompt = RELEVANCY_JUDGE_PROMPT.format(question=question, answer=answer)
    return _parse_json_response(llm.invoke(prompt).content)


def keyword_coverage(answer: str, expected_facts: list) -> float:
    """Fraction of expected facts that appear verbatim in the answer (cheap sanity check)."""
    answer_lower = answer.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return hits / len(expected_facts) if expected_facts else 1.0


def run_eval(api_key: str, k: int = 5) -> dict:
    vector_store = build_vector_store(api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key)
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key)

    rows = []
    for item in EVAL_SET:
        retrieved_ids, retrieved_docs = retrieve_chunk_ids(vector_store, item["question"], k)
        hit = hit_rate(retrieved_ids, item["relevant_chunk_ids"])
        rr = reciprocal_rank(retrieved_ids, item["relevant_chunk_ids"])

        answer = generate_answer(llm, item["question"], retrieved_docs)
        faithfulness = judge_faithfulness(judge, item["question"], retrieved_docs, answer)
        relevancy = judge_relevancy(judge, item["question"], answer)
        coverage = keyword_coverage(answer, item["expected_facts"])

        rows.append({
            "question": item["question"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "hit": hit,
            "reciprocal_rank": rr,
            "answer": answer,
            "faithful": faithfulness.get("faithful"),
            "faithfulness_reasoning": faithfulness.get("reasoning"),
            "relevancy_score": relevancy.get("score"),
            "relevancy_reasoning": relevancy.get("reasoning"),
            "keyword_coverage": coverage,
        })

    report = {
        "k": k,
        "num_questions": len(rows),
        "hit_rate": mean(1.0 if r["hit"] else 0.0 for r in rows),
        "mrr": mean(r["reciprocal_rank"] for r in rows),
        "faithfulness_rate": mean(1.0 if r["faithful"] else 0.0 for r in rows),
        "avg_relevancy_score": mean(r["relevancy_score"] for r in rows),
        "avg_keyword_coverage": mean(r["keyword_coverage"] for r in rows),
        "rows": rows,
    }
    return report


def print_report(report: dict) -> None:
    print(f"\n{'='*72}\nRAG Evaluation Report (k={report['k']}, n={report['num_questions']})\n{'='*72}")
    print(f"  Hit Rate@{report['k']}:          {report['hit_rate']:.1%}")
    print(f"  MRR@{report['k']}:               {report['mrr']:.3f}")
    print(f"  Faithfulness rate:      {report['faithfulness_rate']:.1%}")
    print(f"  Avg relevancy (1-5):    {report['avg_relevancy_score']:.2f}")
    print(f"  Avg keyword coverage:   {report['avg_keyword_coverage']:.1%}")
    print(f"{'-'*72}")
    for row in report["rows"]:
        status = "✅" if row["hit"] else "❌"
        faithful = "✅" if row["faithful"] else "❌"
        print(f"  {status} retrieval | {faithful} faithful | relevancy {row['relevancy_score']}/5 "
              f"| {row['question']}")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Financial RAG Chatbot's retriever and generator.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                         help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve per question")
    parser.add_argument("--output", default="eval_results.json", help="Path to write full JSON results")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: no OpenAI API key provided. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    report = run_eval(args.api_key, k=args.k)
    print_report(report)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
