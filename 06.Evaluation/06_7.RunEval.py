from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
import sys


def _load_local_module(alias: str, filename: str):
    # 번호가 붙은 평가 파일들은 일반 import 대신 파일 경로로 직접 불러온다.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_eval_common = _load_local_module("eval_common", "06_2.EvalCommon.py")
_retrieval_eval = _load_local_module("retrieval_eval", "06_3.RetrievalEval.py")
_faithfulness_eval = _load_local_module("faithfulness_eval", "06_4.FaithfulnessEval.py")
_coverage_eval = _load_local_module("coverage_eval", "06_5.CoverageEval.py")
_rule_eval = _load_local_module("rule_eval", "06_6.RuleEval.py")

DEFAULT_TESTSET_PATH = _eval_common.DEFAULT_TESTSET_PATH
collect_reference_bundle = _eval_common.collect_reference_bundle
generate_plan = _eval_common.generate_plan
get_chain = _eval_common.get_chain
get_vectorstore = _eval_common.get_vectorstore
load_project_modules = _eval_common.load_project_modules
load_test_cases = _eval_common.load_test_cases
save_report = _eval_common.save_report

evaluate_retrieval_case = _retrieval_eval.evaluate_case
evaluate_faithfulness_case = _faithfulness_eval.evaluate_case
evaluate_coverage_case = _coverage_eval.evaluate_case
evaluate_rule_case = _rule_eval.evaluate_case


RULE_CHECK_LABELS = {
    "theory_session_count": "이론 세션 수가 4~6개인지",
    "group_session_count": "그룹 세션이 3개 그룹으로 구성됐는지",
    "each_group_has_2_to_3_sessions": "각 그룹에 세션이 2~3개인지",
    "theory_hours_match": "이론 세션 총 시간이 기준과 맞는지",
    "group_hours_match": "각 그룹 실습 시간이 기준과 맞는지",
    "participant_counts_match": "그룹별 참여 인원 수가 입력값과 맞는지",
}


def _aggregate_scores(case_reports: list[dict]) -> dict:
    retrieval_scores = []
    faithfulness_scores = []
    coverage_scores = []
    rule_scores = []

    for report in case_reports:
        retrieval_scores.append(report["retrieval"]["overall"]["average_precision_at_k"])
        rule_scores.append(report["rule_based"]["score"])

        if "faithfulness" in report:
            faithfulness_scores.append(report["faithfulness"]["score"])
        if "requirement_coverage" in report:
            coverage_scores.append(report["requirement_coverage"]["score"])

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "retrieval_precision_at_k": average(retrieval_scores),
        "faithfulness_score_1_to_5": average(faithfulness_scores),
        "requirement_coverage_score_1_to_5": average(coverage_scores),
        "rule_based_score_0_to_1": average(rule_scores),
    }


def _format_score(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_markdown_report(report: dict) -> str:
    # JSON 원본과 별개로, 사람이 한눈에 읽기 쉬운 한국어 요약 리포트를 만든다.
    lines: list[str] = []
    summary = report["summary"]

    lines.append("# RAG 평가 결과 요약")
    lines.append("")
    lines.append(f"- 테스트셋 경로: `{report['testset_path']}`")
    lines.append(f"- 평가 케이스 수: `{report['case_count']}`")
    lines.append("")
    lines.append("## 전체 요약")
    lines.append("")
    lines.append("| 평가 항목 | 점수 | 설명 |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Retrieval Precision@k | {_format_score(summary['retrieval_precision_at_k'])} | 검색된 상위 문서가 기대 키워드와 얼마나 잘 맞는지 |"
    )
    lines.append(
        f"| Faithfulness | {_format_score(summary['faithfulness_score_1_to_5'])} / 5 | 생성 결과가 검색 문맥에 얼마나 근거하는지 |"
    )
    lines.append(
        f"| Requirement Coverage | {_format_score(summary['requirement_coverage_score_1_to_5'])} / 5 | 사용자 요구사항이 결과에 얼마나 반영됐는지 |"
    )
    lines.append(
        f"| Rule-based Validation | {_format_score(summary['rule_based_score_0_to_1'])} | 시간/세션 수/그룹 구성 규칙을 얼마나 지켰는지 |"
    )

    for case in report["cases"]:
        lines.append("")
        lines.append(f"## 케이스: `{case['case_id']}`")
        lines.append("")
        lines.append("### 1. Retrieval 평가")
        lines.append("")
        lines.append("| 영역 | Precision@k | 해석 |")
        lines.append("|---|---:|---|")
        retrieval = case["retrieval"]
        for section_key, label in (
            ("group_a", "그룹 A 검색"),
            ("group_b", "그룹 B 검색"),
            ("group_c", "그룹 C 검색"),
            ("curriculum_examples", "커리큘럼 예시 검색"),
        ):
            section = retrieval[section_key]
            lines.append(
                f"| {label} | {_format_score(section['precision_at_k'])} | 기대 키워드와 맞는 문서가 `{section['matched_docs']}/{section['k']}`개 포함됨 |"
            )

        lines.append("")
        lines.append("### 2. Rule-based 검사")
        lines.append("")
        lines.append("| 검사 항목 | 결과 | 실제값 | 기대값 |")
        lines.append("|---|---|---|---|")
        for check_key, check in case["rule_based"]["checks"].items():
            label = RULE_CHECK_LABELS.get(check_key, check_key)
            status = "통과" if check["passed"] else "실패"
            lines.append(
                f"| {label} | {status} | `{check['actual']}` | `{check['expected']}` |"
            )

        if "faithfulness" in case:
            faithfulness = case["faithfulness"]
            lines.append("")
            lines.append("### 3. Faithfulness 평가")
            lines.append("")
            lines.append(f"- 점수: `{faithfulness['score']}/5`")
            lines.append(f"- 종합 의견: {faithfulness['summary']}")
            if faithfulness.get("grounded_points"):
                lines.append("- 근거가 잘 보이는 부분:")
                for point in faithfulness["grounded_points"]:
                    lines.append(f"  - {point}")
            if faithfulness.get("unsupported_points"):
                lines.append("- 근거가 부족한 부분:")
                for point in faithfulness["unsupported_points"]:
                    lines.append(f"  - {point}")

        if "requirement_coverage" in case:
            coverage = case["requirement_coverage"]
            lines.append("")
            lines.append("### 4. Requirement Coverage 평가")
            lines.append("")
            lines.append(f"- 점수: `{coverage['score']}/5`")
            lines.append(f"- 종합 의견: {coverage['summary']}")
            if coverage.get("covered_requirements"):
                lines.append("- 잘 반영된 요구사항:")
                for item in coverage["covered_requirements"]:
                    lines.append(f"  - {item}")
            if coverage.get("missing_requirements"):
                lines.append("- 부족하게 반영된 요구사항:")
                for item in coverage["missing_requirements"]:
                    lines.append(f"  - {item}")

    lines.append("")
    return "\n".join(lines)


def _build_timestamped_output_paths() -> tuple[str, str]:
    # 같은 평가를 여러 번 돌릴 수 있으므로, 실행 시각을 파일명에 넣어 결과를 구분한다.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("06.Evaluation", f"{stamp}_EvaluationReport.json")
    markdown_path = os.path.join("06.Evaluation", f"{stamp}_EvaluationReport.md")
    return json_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description="Run the full RAG evaluation suite.")
    parser.add_argument(
        "--testset",
        default=DEFAULT_TESTSET_PATH,
        help="Path to a JSON testset file.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for saving the raw JSON report.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Where to save the Korean markdown report.",
    )
    parser.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help="Skip faithfulness and requirement coverage evaluation.",
    )
    args = parser.parse_args()
    _default_json_output, default_markdown_output = _build_timestamped_output_paths()
    if not args.markdown_output:
        args.markdown_output = default_markdown_output

    modules = load_project_modules()
    vectorstore = get_vectorstore(modules)
    chain = get_chain(modules, vectorstore)
    cases = load_test_cases(args.testset)

    case_reports = []
    for case in cases:
        plan = generate_plan(case, modules, vectorstore, chain)
        case_report = {
            "case_id": case["case_id"],
            "retrieval": evaluate_retrieval_case(case, modules, vectorstore),
            "rule_based": evaluate_rule_case(case, plan, modules),
        }

        if not args.skip_llm_judges:
            references = collect_reference_bundle(case, modules, vectorstore)
            case_report["faithfulness"] = evaluate_faithfulness_case(case, plan, references)
            case_report["requirement_coverage"] = evaluate_coverage_case(case, plan)

        case_reports.append(case_report)

    report = {
        "testset_path": args.testset,
        "case_count": len(case_reports),
        "summary": _aggregate_scores(case_reports),
        "cases": case_reports,
    }
    if args.json_output:
        save_report(report, args.json_output)
    with open(args.markdown_output, "w", encoding="utf-8") as file:
        file.write(_build_markdown_report(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
