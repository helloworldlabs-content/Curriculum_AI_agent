from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def _load_eval_common():
    # 번호가 붙은 파일명은 일반 import가 어려워서, 같은 폴더 파일을 경로로 직접 불러온다.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_2.EvalCommon.py")
    spec = importlib.util.spec_from_file_location("eval_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_common"] = module
    spec.loader.exec_module(module)
    return module


_eval_common = _load_eval_common()
generate_plan = _eval_common.generate_plan
get_chain = _eval_common.get_chain
get_vectorstore = _eval_common.get_vectorstore
load_project_modules = _eval_common.load_project_modules
load_test_cases = _eval_common.load_test_cases
plan_to_text = _eval_common.plan_to_text


JUDGE_MODEL = "gpt-4o-mini"
_COVERAGE_JUDGE = None


class CoverageResult(BaseModel):
    score: int = Field(ge=1, le=5, description="5점에 가까울수록 요구사항 반영도가 높다.")
    covered_requirements: list[str] = Field(description="잘 반영된 요구사항")
    missing_requirements: list[str] = Field(description="부족하게 반영된 요구사항")
    summary: str = Field(description="짧은 종합 의견")


def _get_judge():
    # Coverage 평가는 요구사항과 결과를 직접 비교하는 Judge 모델을 사용한다.
    global _COVERAGE_JUDGE
    if _COVERAGE_JUDGE is None:
        llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        _COVERAGE_JUDGE = llm.with_structured_output(CoverageResult)
    return _COVERAGE_JUDGE


def evaluate_case(case: dict, plan) -> dict:
    judge = _get_judge()
    must_cover = case.get("expected", {}).get("must_cover", [])
    result = judge.invoke(
        [
            (
                "system",
                "You are evaluating requirement coverage for a generated curriculum. "
                "Check whether the curriculum reflects the user's goals, audience, level, time constraints, topic, and any additional must-cover items. "
                "Write every field in Korean. The summary, covered_requirements, and missing_requirements must all be written in Korean.",
            ),
            (
                "human",
                f"""
                [Collected Info]
                {json.dumps(case["collected_info"], ensure_ascii=False, indent=2)}

                [Must Cover Items]
                {json.dumps(must_cover, ensure_ascii=False, indent=2)}

                [Generated Curriculum]
                {plan_to_text(plan)}
                """.strip(),
            ),
        ]
    )
    return result.model_dump()


def main():
    parser = argparse.ArgumentParser(description="Run requirement coverage evaluation with an LLM judge.")
    parser.add_argument("--testset", required=True, help="Path to a JSON testset file.")
    args = parser.parse_args()

    modules = load_project_modules()
    vectorstore = get_vectorstore(modules)
    chain = get_chain(modules, vectorstore)
    cases = load_test_cases(args.testset)

    report = {}
    for case in cases:
        plan = generate_plan(case, modules, vectorstore, chain)
        report[case["case_id"]] = evaluate_case(case, plan)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
