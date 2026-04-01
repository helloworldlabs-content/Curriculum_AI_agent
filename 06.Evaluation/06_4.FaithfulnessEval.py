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
collect_reference_bundle = _eval_common.collect_reference_bundle
generate_plan = _eval_common.generate_plan
get_chain = _eval_common.get_chain
get_vectorstore = _eval_common.get_vectorstore
load_project_modules = _eval_common.load_project_modules
load_test_cases = _eval_common.load_test_cases
plan_to_text = _eval_common.plan_to_text


JUDGE_MODEL = "gpt-4o-mini"
_FAITHFULNESS_JUDGE = None


class FaithfulnessResult(BaseModel):
    score: int = Field(ge=1, le=5, description="5점에 가까울수록 근거성이 높다.")
    verdict: str = Field(description="overall verdict")
    grounded_points: list[str] = Field(description="근거가 잘 보이는 요소")
    unsupported_points: list[str] = Field(description="근거가 부족하거나 과장된 요소")
    summary: str = Field(description="짧은 종합 의견")


def _get_judge():
    # Judge 모델도 한 번만 만들어 여러 케이스에서 재사용한다.
    global _FAITHFULNESS_JUDGE
    if _FAITHFULNESS_JUDGE is None:
        llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        _FAITHFULNESS_JUDGE = llm.with_structured_output(FaithfulnessResult)
    return _FAITHFULNESS_JUDGE


def evaluate_case(case: dict, plan, references: dict[str, str]) -> dict:
    judge = _get_judge()
    result = judge.invoke(
        [
            (
                "system",
                "You are evaluating RAG faithfulness. Score whether the generated curriculum is grounded in the provided references. "
                "Do not judge writing style. Focus on whether the claims, session focus, and recommendations are supported by the references. "
                "Write every field in Korean. The verdict, summary, grounded_points, and unsupported_points must all be written in Korean.",
            ),
            (
                "human",
                f"""
                [User Requirements]
                {json.dumps(case["collected_info"], ensure_ascii=False, indent=2)}

                [Retrieved References]
                {references["combined_text"]}

                [Generated Curriculum]
                {plan_to_text(plan)}
                """.strip(),
            ),
        ]
    )
    return result.model_dump()


def main():
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation with an LLM judge.")
    parser.add_argument("--testset", required=True, help="Path to a JSON testset file.")
    args = parser.parse_args()

    modules = load_project_modules()
    vectorstore = get_vectorstore(modules)
    chain = get_chain(modules, vectorstore)
    cases = load_test_cases(args.testset)

    report = {}
    for case in cases:
        plan = generate_plan(case, modules, vectorstore, chain)
        references = collect_reference_bundle(case, modules, vectorstore)
        report[case["case_id"]] = evaluate_case(case, plan, references)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
