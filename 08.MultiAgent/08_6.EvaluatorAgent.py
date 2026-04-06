import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.evaluator")

# ---------------------------------------------------------------------------
# 스키마 / 헬퍼 로드
# ---------------------------------------------------------------------------

def _load_schemas():
    schemas_path = os.path.join(os.path.dirname(__file__), "08_2.AgentSchemas.py")
    key = "agent_schemas_08_eval"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, schemas_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _load_helpers():
    helpers_path = os.path.join(os.path.dirname(__file__), "08_3.AgentHelpers.py")
    key = "agent_helpers_08"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, helpers_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]

# ---------------------------------------------------------------------------
# 시스템 프롬프트 — 정성 평가(feasibility, condition_fit)만 담당
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = Path(__file__).with_name("08_14.EvaluatorAgentPrompt.txt").read_text(encoding="utf-8").strip()


def _fetch_reference_cases(helpers, topic: str, audience: str, level: str) -> str:
    """
    평가 기준이 될 실제 운영 커리큘럼 사례를 웹에서 검색한다.
    검색 실패 시 빈 문자열을 반환한다 (평가 흐름에는 영향 없음).
    """
    queries = [
        f"{topic} 기업 AI 교육 커리큘럼 실제 사례 {level}",
        f"{audience} 대상 {topic} 교육과정 운영 사례",
    ]
    parts: list[str] = []
    for q in queries:
        try:
            result = helpers.web_search(q, max_results=3)
            if result and "오류" not in result:
                parts.append(result)
        except Exception as err:
            logger.warning("[evaluator] web_search failed for '%s': %s", q, err)
    return "\n\n---\n\n".join(parts) if parts else ""

# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------

def run_evaluator(curriculum: dict, collected_info: dict) -> dict:
    """
    생성된 커리큘럼을 평가한다.

    구조/산술 항목(time_constraint_ok, group_balance_ok)은 코드로 먼저 계산하고,
    정성 항목(feasibility_ok, condition_fit_ok)만 LLM에 위임한다.

    Parameters
    ----------
    curriculum : dict
        CurriculumAgent가 생성한 커리큘럼 (CurriculumPlan.model_dump() 결과)
    collected_info : dict
        InfoCollectorAgent가 수집한 원본 요구사항

    Returns
    -------
    dict
        {
            "passed": bool,
            "feedback": str,
            "reply": str,
            "evaluation": dict,
        }
    """
    schemas = _load_schemas()
    helpers = _load_helpers()

    # ── 1. 코드 기반 구조 검증 ────────────────────────────────────────
    struct = helpers.evaluate_structure(curriculum, collected_info)
    time_constraint_ok = struct["time_constraint_ok"]
    group_balance_ok   = struct["group_balance_ok"]
    time_issues        = struct["time_issues"]
    group_issues       = struct["group_issues"]
    time_summary       = struct["time_summary"]
    active_groups      = struct["active_groups"]

    logger.info(
        "[evaluator] code check — time_ok=%s group_ok=%s time_issues=%s group_issues=%s",
        time_constraint_ok, group_balance_ok, len(time_issues), len(group_issues),
    )

    days          = collected_info.get("days", 1)
    hours_per_day = collected_info.get("hours_per_day", 8)
    topic         = collected_info.get("topic", "")
    audience      = collected_info.get("audience", "")
    level         = collected_info.get("level", "")

    # ── 2. 실제 운영 커리큘럼 사례 웹 검색 ──────────────────────────────
    logger.info("[evaluator] fetching reference cases for feasibility check")
    reference_cases = _fetch_reference_cases(helpers, topic, audience, level)
    if reference_cases:
        logger.info("[evaluator] reference cases fetched (%d chars)", len(reference_cases))
    else:
        logger.warning("[evaluator] no reference cases found, feasibility check proceeds without external data")

    # ── 3. LLM 정성 평가 ─────────────────────────────────────────────
    reference_section = (
        f"[실제 운영 커리큘럼 사례 — feasibility 판단 시 비교 기준으로 활용]\n{reference_cases}"
        if reference_cases
        else "[실제 사례 검색 결과 없음 — 일반적인 기업 교육 기준으로 판단]"
    )

    prompt = dedent(
        f"""
        아래 커리큘럼의 강의 실행 가능성(feasibility_ok)과 요구사항 반영 여부(condition_fit_ok)를 평가해 주세요.

        [요구사항]
        회사명: {collected_info.get('company_name', '')}
        교육 목표: {collected_info.get('goal', '')}
        교육 대상: {audience}
        AI 활용 수준: {level}
        교육 기간: {days}일 / 하루 {hours_per_day}시간
        핵심 주제: {topic}
        제약 사항: {collected_info.get('constraints', '')}
        실제 인원이 있는 그룹: {', '.join(active_groups) if active_groups else '없음'}

        [시간 합계 검증 결과 — 코드로 계산 완료, 재평가 불필요]
        {time_summary}

        {reference_section}

        [생성된 커리큘럼]
        ```json
        {json.dumps(curriculum, ensure_ascii=False, indent=2)}
        ```
        """
    ).strip()

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        llm_result: schemas.LLMEvaluationResult = llm.with_structured_output(
            schemas.LLMEvaluationResult
        ).invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

        feasibility_score = llm_result.feasibility_score  # -1이면 사례 비교 모드
        condition_fit_ok  = llm_result.condition_fit_ok
        summary           = llm_result.summary

        # 사례 없는 경우: 점수 기반으로 feasibility_ok 재판정
        if not reference_cases and feasibility_score >= 0:
            feasibility_ok = feasibility_score >= 7
            logger.info(
                "[evaluator] no reference cases — score-based feasibility: score=%s ok=%s",
                feasibility_score, feasibility_ok,
            )
        else:
            feasibility_ok = llm_result.feasibility_ok

        # ── 3. 피드백 조립 — 실패 항목만, 확인된 근거만 ───────────────
        feedback_parts: list[str] = []
        if not time_constraint_ok:
            feedback_parts.append("## 시간 조정 필요\n" + "\n".join(f"- {i}" for i in time_issues))
        if not group_balance_ok:
            feedback_parts.append("## 그룹 세션 누락 수정 필요\n" + "\n".join(f"- {i}" for i in group_issues))
        if not feasibility_ok and llm_result.feasibility_feedback:
            feedback_parts.append("## 세션 실행 가능성 개선 필요\n" + llm_result.feasibility_feedback)
        if not condition_fit_ok and llm_result.condition_feedback:
            feedback_parts.append("## 요구사항 반영 보완 필요\n" + llm_result.condition_feedback)

        feedback = "\n\n".join(feedback_parts)

        # ── 4. 최종 passed 판정 ───────────────────────────────────────
        passed = time_constraint_ok and group_balance_ok and feasibility_ok and condition_fit_ok

        evaluation = {
            "time_constraint_ok": time_constraint_ok,
            "feasibility_ok":     feasibility_ok,
            "feasibility_score":  feasibility_score,  # -1: 사례 비교, 0~10: 자체 점수
            "condition_fit_ok":   condition_fit_ok,
            "group_balance_ok":   group_balance_ok,
            "passed":             passed,
            "feedback":           feedback,
            "summary":            summary,
        }

        logger.info(
            "[evaluator] passed=%s time=%s feasibility=%s condition=%s balance=%s",
            passed, time_constraint_ok, feasibility_ok, condition_fit_ok, group_balance_ok,
        )
        return {
            "passed":     passed,
            "feedback":   feedback,
            "reply":      summary,
            "evaluation": evaluation,
        }

    except Exception as err:
        logger.error("[evaluator] LLM evaluation failed: %s", err)
        # LLM 실패 시 코드 검증 결과만으로 판정
        feedback_parts = []
        if not time_constraint_ok:
            feedback_parts.append("## 시간 조정 필요\n" + "\n".join(f"- {i}" for i in time_issues))
        if not group_balance_ok:
            feedback_parts.append("## 그룹 세션 누락 수정 필요\n" + "\n".join(f"- {i}" for i in group_issues))

        passed   = time_constraint_ok and group_balance_ok
        feedback = "\n\n".join(feedback_parts)
        summary  = "평가 중 오류가 발생해 구조 검증 결과만 반영합니다." if not passed else "구조 검증을 통과했습니다."

        return {
            "passed":     passed,
            "feedback":   feedback,
            "reply":      summary,
            "evaluation": {
                "time_constraint_ok": time_constraint_ok,
                "feasibility_ok":     True,
                "condition_fit_ok":   True,
                "group_balance_ok":   group_balance_ok,
                "passed":             passed,
            },
        }
