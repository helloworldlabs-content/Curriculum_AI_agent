from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def _load_eval_common():
    # 번호가 붙은 파일명은 일반 import가 어려워서, 같은 폴더 파일을 경로로 직접 불러온다.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_2.EvalCommon.py")
    spec = importlib.util.spec_from_file_location("eval_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_common"] = module
    spec.loader.exec_module(module)
    return module


_eval_common = _load_eval_common()
build_case_inputs = _eval_common.build_case_inputs
get_vectorstore = _eval_common.get_vectorstore
load_project_modules = _eval_common.load_project_modules
load_test_cases = _eval_common.load_test_cases


def _keyword_precision_at_k(docs, expected_keywords: list[str], *, k: int) -> dict:
    # 골든 문서 ID가 없을 때는, 기대 키워드가 상위 문서에 얼마나 들어갔는지로 간단히 Precision@k를 본다.
    top_docs = docs[:k]
    lowered_keywords = [keyword.lower() for keyword in expected_keywords if keyword]
    if not top_docs:
        return {"precision_at_k": 0.0, "matched_docs": 0, "k": k}

    matched_docs = 0
    for doc in top_docs:
        haystack = f"{doc.page_content} {json.dumps(doc.metadata, ensure_ascii=False)}".lower()
        if any(keyword in haystack for keyword in lowered_keywords):
            matched_docs += 1

    return {
        "precision_at_k": matched_docs / len(top_docs),
        "matched_docs": matched_docs,
        "k": len(top_docs),
    }


def evaluate_case(case: dict, modules=None, vectorstore=None) -> dict:
    modules = modules or load_project_modules()
    vectorstore = vectorstore or get_vectorstore(modules)
    retrieval = modules.retrieval
    info, groups, _conversation = build_case_inputs(case, modules)
    expected = case.get("expected", {}).get("retrieval", {})
    corpus_cache: dict = {}

    group_k = int(expected.get("group_k", 4))
    curriculum_k = int(expected.get("curriculum_k", 3))

    results = {}
    for group_key in ("group_a", "group_b", "group_c"):
        group = groups[group_key]
        docs = retrieval._retrieve(
            vectorstore,
            f"{', '.join(group['types'])} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법",
            k=group_k,
            search_filter={
                "$and": [
                    {"doc_type": {"$eq": "ax_compass"}},
                    {"type_name": {"$in": group["types"]}},
                ]
            },
            label=f"{group_key}_eval",
            corpus_cache=corpus_cache,
        )
        expected_keywords = expected.get(f"{group_key}_keywords", group["types"])
        results[group_key] = {
            **_keyword_precision_at_k(docs, expected_keywords, k=group_k),
            "expected_keywords": expected_keywords,
        }

    curriculum_docs = retrieval._retrieve(
        vectorstore,
        retrieval._build_curriculum_query(info),
        k=curriculum_k,
        search_filter={"doc_type": {"$eq": "curriculum_example"}},
        label="curriculum_examples_eval",
        corpus_cache=corpus_cache,
    )
    curriculum_keywords = expected.get(
        "curriculum_keywords",
        [info.topic, info.goal, info.audience, info.level],
    )
    results["curriculum_examples"] = {
        **_keyword_precision_at_k(curriculum_docs, curriculum_keywords, k=curriculum_k),
        "expected_keywords": curriculum_keywords,
    }

    score_values = [section["precision_at_k"] for section in results.values()]
    results["overall"] = {
        "average_precision_at_k": sum(score_values) / len(score_values) if score_values else 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run retrieval quality evaluation.")
    parser.add_argument("--testset", required=True, help="Path to a JSON testset file.")
    args = parser.parse_args()

    modules = load_project_modules()
    vectorstore = get_vectorstore(modules)
    cases = load_test_cases(args.testset)
    report = {case["case_id"]: evaluate_case(case, modules, vectorstore) for case in cases}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
