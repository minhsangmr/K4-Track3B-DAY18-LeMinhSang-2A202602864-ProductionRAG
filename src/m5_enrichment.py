from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


import json as _json
import re


def _get_llm_client():
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        return None, None
    try:
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return client, model
    except Exception:
        return None, None


def _call_llm(messages, max_tokens=300):
    client, model = _get_llm_client()
    if not client:
        return None
    models_to_try = [model, "ag/gemini-3.8-flash", "gpt-4o-mini"]
    seen = set()
    models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception:
            continue
    return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""

    llm_resp = _call_llm([
        {"role": "system", "content": "Tóm tắt đoạn văn sau cực kỳ ngắn gọn trong đúng 1 câu ngắn bằng tiếng Việt, dưới 20 từ. Không thêm lời dẫn."},
        {"role": "user", "content": text},
    ], max_tokens=60)
    if llm_resp:
        if len(llm_resp) > len(text) * 1.8:
            sentences = [s.strip() for s in llm_resp.split(". ") if s.strip()]
            if sentences and len(sentences[0]) <= len(text) * 1.8:
                return sentences[0] + ("." if not sentences[0].endswith(".") else "")
            return llm_resp[: int(len(text) * 1.8)].rstrip()
        return llm_resp

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip():
        return []

    llm_resp = _call_llm([
        {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
        {"role": "user", "content": text},
    ], max_tokens=200)
    if llm_resp:
        questions = llm_resp.split("\n")
        parsed = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
        if parsed:
            return parsed[:n_questions]

    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return ""

    llm_resp = _call_llm([
        {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
        {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=80)
    if llm_resp:
        return f"{llm_resp}\n\n{text}"

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if not text.strip():
        return {}

    llm_resp = _call_llm([
        {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
        {"role": "user", "content": text},
    ], max_tokens=150)
    if llm_resp:
        try:
            cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", llm_resp, flags=re.MULTILINE).strip()
            return _json.loads(cleaned)
        except Exception:
            pass

    return {"topic": "quy chế", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if not text.strip():
        return {}

    prompt = f"""Phân tích đoạn văn và trả về JSON:
{{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}
}}"""

    llm_resp = _call_llm([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=400)

    if llm_resp:
        try:
            cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", llm_resp, flags=re.MULTILINE).strip()
            parsed = _json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Fallback chất lượng cao
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu {source}." if source else "Trích từ quy chế nội bộ.",
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    from concurrent.futures import ThreadPoolExecutor

    def _process_one(item):
        i, chunk = item
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        enriched = list(executor.map(_process_one, enumerate(chunks)))

    print(f"  Enriched {len(enriched)}/{len(chunks)} chunks.", flush=True)
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
