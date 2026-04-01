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
generate_plan = _eval_common.generate_plan
get_chain = _eval_common.get_chain
get_vectorstore = _eval_common.get_vectorstore
load_project_modules = _eval_common.load_project_modules
load_test_cases = _eval_common.load_test_cases


def _sum_session_hours(sessions) -> float:
    return round(sum(session.duration_hours for session in sessions), 2)


def _close_enough(actual: float, expected: float, tolerance: float = 0.25) -> bool:
    # 시간은 소수점 반올림 차이가 생길 수 있어서 약간의 오차를 허용한다.
    return abs(actual - expected) <= tolerance


def evaluate_case(case: dict, plan, modules=None) -> dict:
    modules = modules or load_project_modules()
    info, groups, _conversation = build_case_inputs(case, modules)
    total_hours = info.days * info.hours_per_day
    expected_theory_hours = round(total_hours * 0.65)
    expected_group_hours = total_hours - expected_theory_hours

    actual_theory_hours = _sum_session_hours(plan.theory_sessions)
    actual_group_hours = [_sum_session_hours(group.sessions) for group in plan.group_sessions]

    checks = {
        "theory_session_count": {
            "passed": 4 <= len(plan.theory_sessions) <= 6,
            "actual": len(plan.theory_sessions),
            "expected": "between 4 and 6",
        },
        "group_session_count": {
            "passed": len(plan.group_sessions) == 3,
            "actual": len(plan.group_sessions),
            "expected": 3,
        },
        "each_group_has_2_to_3_sessions": {
            "passed": all(2 <= len(group.sessions) <= 3 for group in plan.group_sessions),
            "actual": [len(group.sessions) for group in plan.group_sessions],
            "expected": "each group between 2 and 3",
        },
        "theory_hours_match": {
            "passed": _close_enough(actual_theory_hours, expected_theory_hours),
            "actual": actual_theory_hours,
            "expected": expected_theory_hours,
        },
        "group_hours_match": {
            "passed": all(_close_enough(hours, expected_group_hours) for hours in actual_group_hours),
            "actual": actual_group_hours,
            "expected": expected_group_hours,
        },
        "participant_counts_match": {
            "passed": sorted(group.participant_count for group in plan.group_sessions)
            == sorted(group_info["count"] for group_info in groups.values()),
            "actual": sorted(group.participant_count for group in plan.group_sessions),
            "expected": sorted(group_info["count"] for group_info in groups.values()),
        },
    }

    passed_checks = sum(1 for check in checks.values() if check["passed"])
    return {
        "passed_checks": passed_checks,
        "total_checks": len(checks),
        "score": passed_checks / len(checks) if checks else 0.0,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Run rule-based curriculum validation.")
    parser.add_argument("--testset", required=True, help="Path to a JSON testset file.")
    args = parser.parse_args()

    modules = load_project_modules()
    vectorstore = get_vectorstore(modules)
    chain = get_chain(modules, vectorstore)
    cases = load_test_cases(args.testset)

    report = {}
    for case in cases:
        plan = generate_plan(case, modules, vectorstore, chain)
        report[case["case_id"]] = evaluate_case(case, plan, modules)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
