from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


_LAST_LATENCIES: dict[str, float] = {}


def build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker]:
    """Build production RAG pipeline and track latencies."""
    global _LAST_LATENCIES
    _LAST_LATENCIES = {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    _LAST_LATENCIES["m1_chunking"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({_LAST_LATENCIES['m1_chunking']}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        _LAST_LATENCIES["m5_enrichment"] = round(time.time() - t0, 2)
        print(f"  ✓ Enriched {len(enriched)} chunks ({_LAST_LATENCIES['m5_enrichment']}s)", flush=True)
    else:
        _LAST_LATENCIES["m5_enrichment"] = 0.0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    _LAST_LATENCIES["m2_indexing"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({_LAST_LATENCIES['m2_indexing']}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    _LAST_LATENCIES["m3_reranker_loading"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({_LAST_LATENCIES['m3_reranker_loading']}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            system_prompt = (
                "Bạn là trợ lý AI chuyên nghiệp về quy chế nội bộ. "
                "Hãy trả lời câu hỏi CHỈ dựa trên thông tin trong Context được cung cấp. "
                "Trả lời trực tiếp, chính xác, ngắn gọn, đầy đủ ý, không suy diễn hoặc thêm thông tin ngoài context. "
                "Nếu trong context không có thông tin, hãy trả lời: 'Không tìm thấy.'"
            )
            models = [os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "ag/gemini-3.8-flash", "gpt-4o-mini"]
            answer = None
            seen = set()
            for m in models:
                if not m or m in seen:
                    continue
                seen.add(m)
                try:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    content = resp.choices[0].message.content
                    if content and content.strip():
                        answer = content.strip()
                        break
                except Exception:
                    continue
            if not answer:
                answer = contexts[0]
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, latencies: dict[str, float] | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    latencies = latencies if latencies is not None else _LAST_LATENCIES.copy()

    t_gen_start = time.time()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    latencies["query_and_generation"] = round(time.time() - t_gen_start, 2)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    latencies["m4_ragas_eval"] = round(time.time() - t0, 2)
    print(f"  ✓ RAGAS done ({latencies['m4_ragas_eval']}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency Breakdown Report
    total_time = sum(latencies.values())
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN REPORT")
    print("-" * 60)
    print(f"{'Stage':<35} {'Time (s)':>10} {'Pct (%)':>10}")
    print("-" * 60)
    for stage, t in latencies.items():
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"{stage:<35} {t:>10.2f} {pct:>9.1f}%")
    print("-" * 60)
    print(f"{'Total Pipeline Latency':<35} {total_time:>10.2f} {100.0:>9.1f}%")
    print("=" * 60)

    failures = failure_analysis(results.get("per_question", []))
    results["latency_breakdown"] = latencies
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
