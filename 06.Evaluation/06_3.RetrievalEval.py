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


def _keyword_hit_rate(text: str, keywords: list[str]) -> dict:
    """
    반환된 텍스트에 기대 키워드가 몇 개나 포함되는지 계산한다.

    이전 _keyword_precision_at_k 는 Document 목록 기준이었으나,
    이제 retrieve_group_context / retrieve_curriculum_examples 가 반환하는
    문자열(string)을 직접 검사하는 방식으로 변경한다.

    - matched_docs  : 키워드 중 텍스트에 포함된 키워드 수
    - k             : 기대 키워드 총 수 (기존 API 호환을 위해 k 로 표기)
    - precision_at_k: matched_docs / k
    """
    if not text or not keywords:
        return {"precision_at_k": 0.0, "matched_docs": 0, "k": len(keywords)}

    lowered = text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in lowered)
    return {
        "precision_at_k": round(matched / len(keywords), 2),
        "matched_docs": matched,
        "k": len(keywords),
    }


def evaluate_case(case: dict, modules=None, vectorstore=None) -> dict:
    """
    프로덕션 retrieval 함수(retrieve_group_context, retrieve_curriculum_examples)를
    그대로 호출해 평가한다.

    [기존 방식의 문제]
    내부 _retrieve 함수를 type_name 필터로 직접 호출하면,
    PDF 청크에 type_name 메타데이터가 없을 때 결과가 0건이 되어
    Precision 이 항상 0.00 으로 나온다.

    [수정 방식]
    실제 서비스와 동일한 함수(retrieve_group_context 등)를 호출하므로,
    평가 결과가 실제 서비스 품질을 정확히 반영한다.
    type_name 필터 문제가 있더라도 그 자체를 평가하는 것이 목적이므로
    평가 로직은 서비스 코드와 동일한 경로여야 한다.
    """
    modules = modules or load_project_modules()
    vectorstore = vectorstore or get_vectorstore(modules)
    retrieval = modules.retrieval
    info, groups, _conversation = build_case_inputs(case, modules)
    expected = case.get("expected", {}).get("retrieval", {})
    corpus_cache: dict = {}

    results = {}
    for group_key in ("group_a", "group_b", "group_c"):
        group = groups[group_key]
        # 프로덕션과 동일한 함수 호출: type_name 필터 + Hybrid Search + Rerank
        ctx_text = retrieval.retrieve_group_context(
            vectorstore,
            group["types"],
            group_label=f"{group_key}_eval",
            corpus_cache=corpus_cache,
        )
        expected_keywords = expected.get(f"{group_key}_keywords", group["types"])
        results[group_key] = {
            **_keyword_hit_rate(ctx_text, expected_keywords),
            "expected_keywords": expected_keywords,
            "retrieved_chars": len(ctx_text),
        }

    # 커리큘럼 예시 검색도 프로덕션 함수 그대로 사용
    curriculum_text = retrieval.retrieve_curriculum_examples(
        vectorstore,
        info,
        corpus_cache=corpus_cache,
    )
    curriculum_keywords = expected.get(
        "curriculum_keywords",
        [info.topic, info.goal, info.audience, info.level],
    )
    results["curriculum_examples"] = {
        **_keyword_hit_rate(curriculum_text, curriculum_keywords),
        "expected_keywords": curriculum_keywords,
        "retrieved_chars": len(curriculum_text),
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
