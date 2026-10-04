from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        from config import OPENAI_API_KEY
        import math
        kwargs = {}
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from langchain_community.embeddings import HuggingFaceEmbeddings

                model_name = os.getenv("OPENAI_MODEL", "ag/gemini-3.8-flash")
                base_url = os.getenv("OPENAI_BASE_URL")
                raw_llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_base=base_url,
                    temperature=0.0,
                )
                ragas_llm = LangchainLLMWrapper(raw_llm)
                kwargs["llm"] = ragas_llm

                raw_emb = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )
                ragas_emb = LangchainEmbeddingsWrapper(raw_emb)
                kwargs["embeddings"] = ragas_emb

                for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    m.llm = ragas_llm
                    if hasattr(m, "embeddings"):
                        m.embeddings = ragas_emb
            except Exception as ex:
                print(f"  ⚠️  Setting up custom RAGAS LLM/Embeddings failed: {ex}")

        def _val(x):
            try:
                v = float(x)
                return 0.0 if math.isnan(v) else v
            except Exception:
                return 0.0

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            **kwargs
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_val(row.get("faithfulness")),
                answer_relevancy=_val(row.get("answer_relevancy")),
                context_precision=_val(row.get("context_precision")),
                context_recall=_val(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _val(result.get("faithfulness")),
            "answer_relevancy": _val(result.get("answer_relevancy")),
            "context_precision": _val(result.get("context_precision")),
            "context_recall": _val(result.get("context_recall")),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_q = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_q,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu",
            "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
            "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25",
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu",
            "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata",
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn",
        ),
    }

    if not eval_results:
        return []

    failures = []
    for r in eval_results:
        metrics_dict = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(metrics_dict.values()) / 4.0
        worst_metric = min(metrics_dict, key=metrics_dict.get)
        worst_score = metrics_dict[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown error", "Review pipeline")
        )
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda x: x["avg_score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
