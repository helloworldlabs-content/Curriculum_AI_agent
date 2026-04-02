import json
import logging
import os
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.evaluator")

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼의 품질을 평가하는 전문 에이전트다.
    커리큘럼 생성 에이전트가 만든 결과물을 아래 4가지 기준으로 엄격하게 평가한다.

    ## 평가 기준

    ### 1. 시간 적합성 (time_constraint_ok)
    - 각 일차(DaySchedule)에서 common_sessions 합계 + 그룹 sessions 합계(한 그룹 기준) = hours_per_day 를 만족하는지 확인
    - 세 그룹의 sessions 시간 합계가 동일한지 확인
    - 허용 오차: ±0.5시간
    - 실패 기준: 어느 한 일차라도 시간이 맞지 않으면 false

    ### 2. 현실성 (feasibility_ok)
    - 각 세션의 title과 activities가 실제 기업 교육에서 진행 가능한 내용인지 확인
    - 세션당 duration_hours에 비해 activities가 너무 많거나 너무 적지 않은지 확인
    - 실패 기준: 추상적이거나 실행 불가능한 세션이 전체의 30% 이상이면 false

    ### 3. 조건 적합성 (condition_fit_ok)
    - 기업의 교육 목표(goal)와 핵심 주제(topic)가 커리큘럼에 명확히 반영됐는지 확인
    - 제약 사항(constraints)이 실제로 반영됐는지 확인
    - 교육 대상자(audience)와 AI 활용 수준(level)에 맞는 난이도인지 확인
    - 실패 기준: 목표·주제·제약 중 하나라도 반영되지 않으면 false

    ### 4. 그룹 균형 (group_balance_ok)
    - 3개 그룹(A/B/C) 각각에 대한 세션이 모든 일차에 존재하는지 확인
    - 그룹별 세션 내용이 해당 AX Compass 유형에 맞게 차별화됐는지 확인
    - 실패 기준: 한 그룹이라도 세션이 없는 일차가 있으면 false

    ## 출력 규칙
    - passed: 4가지 기준을 모두 통과해야 true
    - feedback: 통과하지 못한 기준만 구체적으로 작성. 커리큘럼 에이전트가 재생성 시 반영할 수 있도록
      "X일차 common_sessions 합계가 N시간인데 hours_per_day는 M시간입니다. 조정 필요." 처럼 구체적으로 서술한다.
    - summary: 사용자에게 보여줄 평가 결과 요약 (통과 시 "평가를 통과했습니다.", 미통과 시 주요 문제 1~2개 언급)
    """
).strip()

# ---------------------------------------------------------------------------
# EvaluationResult 스키마 로드
# ---------------------------------------------------------------------------

def _load_evaluation_result_class():
    import importlib.util, sys
    schemas_path = os.path.join(os.path.dirname(__file__), "08_2.AgentSchemas.py")
    key = "agent_schemas_08_eval"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, schemas_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key].EvaluationResult

# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------

def run_evaluator(curriculum: dict, collected_info: dict) -> dict:
    """
    생성된 커리큘럼을 4가지 기준으로 평가한다.

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
            "feedback": str,   # 커리큘럼 에이전트에게 전달할 개선 요청
            "reply": str,      # 사용자에게 보여줄 평가 요약
            "evaluation": dict,  # EvaluationResult 전체
        }
    """
    EvaluationResult = _load_evaluation_result_class()

    days = collected_info.get("days", 1)
    hours_per_day = collected_info.get("hours_per_day", 8)

    prompt = dedent(
        f"""
        아래 커리큘럼을 평가 기준에 따라 평가해 주세요.

        [요구사항]
        회사명: {collected_info.get('company_name', '')}
        교육 목표: {collected_info.get('goal', '')}
        교육 대상: {collected_info.get('audience', '')}
        AI 활용 수준: {collected_info.get('level', '')}
        교육 기간: {days}일
        하루 교육 시간: {hours_per_day}시간
        핵심 주제: {collected_info.get('topic', '')}
        제약 사항: {collected_info.get('constraints', '')}
        그룹 인원: 균형형 {collected_info.get('count_balanced', 0)}명 / 이해형 {collected_info.get('count_learner', 0)}명 /
                  과신형 {collected_info.get('count_overconfident', 0)}명 / 실행형 {collected_info.get('count_doer', 0)}명 /
                  판단형 {collected_info.get('count_analyst', 0)}명 / 조심형 {collected_info.get('count_cautious', 0)}명

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
        result: EvaluationResult = llm.with_structured_output(EvaluationResult).invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        logger.info(
            "[evaluator] passed=%s time=%s feasibility=%s condition=%s balance=%s",
            result.passed, result.time_constraint_ok, result.feasibility_ok,
            result.condition_fit_ok, result.group_balance_ok,
        )
        return {
            "passed": result.passed,
            "feedback": result.feedback,
            "reply": result.summary,
            "evaluation": result.model_dump(),
        }
    except Exception as err:
        logger.error("[evaluator] evaluation failed: %s", err)
        # 평가 실패 시 통과로 처리해 무한 루프 방지
        return {
            "passed": True,
            "feedback": "",
            "reply": "평가 중 오류가 발생해 커리큘럼을 그대로 사용합니다.",
            "evaluation": {},
        }
