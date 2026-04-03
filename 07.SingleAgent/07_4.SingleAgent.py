import importlib.util
import logging
import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("single_agent.agent")

PROMPT_PATH = Path(__file__).with_name("07_10.SystemPrompt.txt")


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_system_prompt()


def _load_helpers():
    helpers_path = os.path.join(os.path.dirname(__file__), "07_3.AgentHelpers.py")
    if "agent_helpers" not in sys.modules:
        spec = importlib.util.spec_from_file_location("agent_helpers", helpers_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent_helpers"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["agent_helpers"]


def _build_tools(vectorstore: Chroma, curriculum_holder: dict, info_dict_holder: dict) -> list:
    """
    vectorstore, curriculum_holder, info_dict_holder를 클로저로 캡처한 LangChain 도구 목록을 반환한다.
    요청마다 새 그래프를 만들 때 호출하므로 매번 새 도구 객체가 생성된다.
    """
    helpers = _load_helpers()

    @tool
    def web_search(query: str) -> str:
        """
        Tavily API로 웹을 검색한다.
        최신 AI 교육 트렌드, 기업 도입 사례, 실제 커리큘럼 구성 사례 등을 파악할 때 사용한다.
        수집된 정보와 목적에 따라 필요한 횟수만큼 자율적으로 호출한다.
        """
        return helpers.web_search(query)

    @tool
    def search_ax_compass(query: str, type_names: list[str] | None = None) -> str:
        """
        내부 VectorDB에서 AX Compass 유형별 특성, 강점, 보완 방향, 교육 접근법을 검색한다.
        type_names에 실제 인원이 있는 그룹의 유형만 지정한다. 인원이 0명인 그룹의 유형은 생략 가능하다.
        예: 그룹 A(균형형·이해형)만 검색 시 type_names=["균형형", "이해형"]
        """
        return helpers.retrieve_ax_compass(vectorstore, query, type_names or None)

    @tool
    def search_curriculum_examples(query: str) -> str:
        """
        내부 VectorDB에서 기업 AI 교육 커리큘럼 예시를 검색한다.
        사용자 요구사항과 유사한 기존 커리큘럼 사례를 확인할 때 사용한다.
        """
        return helpers.retrieve_curriculum_examples(vectorstore, query)

    @tool
    def generate_curriculum(
        company_name: str,
        goal: str,
        audience: str,
        level: str,
        days: int,
        hours_per_day: int,
        topic: str,
        constraints: str,
        count_balanced: int,
        count_learner: int,
        count_overconfident: int,
        count_doer: int,
        count_analyst: int,
        count_cautious: int,
        ax_context_a: str,
        ax_context_b: str,
        ax_context_c: str,
        curriculum_context: str,
        web_context: str,
    ) -> str:
        """
        수집한 정보와 검색 결과를 바탕으로 일차 단위 커리큘럼을 생성한다.
        호출 전에 필요한 검색(search_ax_compass, search_curriculum_examples, web_search)을
        자율적으로 판단하여 충분히 수행한 뒤 호출한다.
        호출 후에는 반드시 validate_curriculum을 호출하여 결과를 검증해야 한다.

        ax_context_a: 그룹 A(균형형·이해형) AX Compass 검색 결과 (인원 없으면 빈 문자열)
        ax_context_b: 그룹 B(과신형·실행형) AX Compass 검색 결과 (인원 없으면 빈 문자열)
        ax_context_c: 그룹 C(판단형·조심형) AX Compass 검색 결과 (인원 없으면 빈 문자열)
        curriculum_context: 내부 커리큘럼 예시 검색 결과
        web_context: 웹 검색으로 수집한 사례 및 최신 트렌드
        """
        info_dict = {
            "company_name": company_name,
            "goal": goal,
            "audience": audience,
            "level": level,
            "days": days,
            "hours_per_day": hours_per_day,
            "topic": topic,
            "constraints": constraints,
            "count_balanced": count_balanced,
            "count_learner": count_learner,
            "count_overconfident": count_overconfident,
            "count_doer": count_doer,
            "count_analyst": count_analyst,
            "count_cautious": count_cautious,
        }
        curriculum = helpers.generate_curriculum(
            info_dict,
            ax_context_a=ax_context_a,
            ax_context_b=ax_context_b,
            ax_context_c=ax_context_c,
            curriculum_context=curriculum_context,
            web_context=web_context,
        )
        MAX_REGEN = 3  # 원본 1회 + 재생성 최대 2회
        current_count = curriculum_holder.get("regen_count", 0)
        if current_count >= MAX_REGEN:
            return (
                f"[생성 차단] 이미 {current_count}회 생성하여 상한({MAX_REGEN}회)에 도달했습니다. "
                "더 이상 generate_curriculum을 호출하지 마세요. "
                "현재 저장된 결과와 검증 실패 사유를 사용자에게 안내하세요."
            )

        regen_count = current_count + 1
        curriculum_holder["result"] = curriculum
        curriculum_holder["regen_count"] = regen_count
        curriculum_holder["validated"] = False
        curriculum_holder["validation_attempted"] = False
        info_dict_holder["info"] = info_dict
        return (
            f"커리큘럼이 생성되었습니다 (생성 {regen_count}/{MAX_REGEN}회차). "
            "이제 validate_curriculum을 호출하여 구조 검증을 수행하세요."
        )

    @tool
    def validate_curriculum() -> str:
        """
        가장 최근에 generate_curriculum으로 생성된 커리큘럼의 구조적 무결성을 검사한다.
        generate_curriculum 호출 직후 반드시 이 도구를 호출해야 한다.

        검사 항목:
        - 시간 무결성: 각 일차의 common + group 합계 == hours_per_day
        - 그룹 커버리지: 인원이 있는 그룹이 그룹 세션에 모두 포함되었는지
        - 세션 타입 규칙: 공통 세션 ↔ '공통 이론'/'공통 실습', 그룹 세션 ↔ '그룹별 프로젝트'/'그룹별 심화 적용'
        - 전체 구조: 일수 일치, program_title 존재
        - 내용 충실도: goals/contents 3개 이상 (Warning만 표시, 재생성 불필요)

        반환: "PASS" 또는 "FAIL: [문제 목록]"
        FAIL이고 재생성 가능 횟수가 남아있으면 generate_curriculum을 재호출하여 수정한다.
        재생성 횟수 상한(3회)에 도달하면 더 이상 재생성하지 말고 현재 결과와 문제점을 사용자에게 안내한다.
        """
        curriculum = curriculum_holder.get("result")
        info = info_dict_holder.get("info")
        if curriculum is None:
            return "검증할 커리큘럼이 없습니다. generate_curriculum을 먼저 호출하세요."
        if info is None:
            return "info_dict가 없습니다. generate_curriculum을 먼저 호출하세요."

        curriculum_holder["validation_attempted"] = True
        result = helpers.validate_curriculum_result(curriculum, info)

        if result.startswith("PASS"):
            curriculum_holder["validated"] = True
            return result

        curriculum_holder["validated"] = False
        regen_count = curriculum_holder.get("regen_count", 1)
        max_regen = 3  # 원본 1회 + 재생성 최대 2회
        remaining = max_regen - regen_count

        if remaining <= 0:
            return (
                result
                + "\n\n[재생성 횟수 상한 도달: 추가 재생성 금지]"
                " 현재 결과를 사용자에게 전달하고 위 문제점을 함께 안내하세요."
            )
        return (
            result
            + f"\n\n[현재 생성 {regen_count}회차 / 상한 {max_regen}회 | 재생성 가능 {remaining}회 남음]"
            " constraints 파라미터에 위 문제 항목을 명시하고 generate_curriculum을 재호출하세요."
        )

    return [web_search, search_ax_compass, search_curriculum_examples, generate_curriculum, validate_curriculum]


def run_agent(
    messages: list[dict],
    vectorstore: Chroma,
    *,
    max_iterations: int = 25,
) -> dict:
    """
    LangGraph ReAct 에이전트를 실행한다.

    Parameters
    ----------
    messages : list[dict]
        클라이언트가 전달한 대화 기록 {"role": "user"|"assistant", "content": str}
    vectorstore : Chroma
        RAG에 사용할 벡터스토어
    max_iterations : int
        recursion_limit = max_iterations * 3 (자기 평가 루프 포함)

    Returns
    -------
    dict
        {"reply": str, "complete": bool, "curriculum": dict | None}
    """
    curriculum_holder: dict = {}
    info_dict_holder: dict = {}

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tools = _build_tools(vectorstore, curriculum_holder, info_dict_holder)

    graph = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)

    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]

    logger.info("[agent] invoke start messages=%s", len(lc_messages))

    try:
        result = graph.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": max_iterations * 3},
        )
        last = result["messages"][-1]
        reply = last.content if isinstance(last.content, str) else str(last.content)

        # Fix 1: 에이전트가 validate_curriculum을 호출하지 않은 경우 강제 실행
        if "result" in curriculum_holder and not curriculum_holder.get("validation_attempted"):
            info = info_dict_holder.get("info")
            if info:
                logger.warning("[agent] validate_curriculum 미호출 감지 — 강제 검증 실행")
                forced = helpers.validate_curriculum_result(curriculum_holder["result"], info)
                curriculum_holder["validation_attempted"] = True
                curriculum_holder["validated"] = forced.startswith("PASS")
                curriculum_holder["forced_validation"] = forced
                logger.info("[agent] 강제 검증 결과: %s", forced[:120])

        has_curriculum = "result" in curriculum_holder
        is_validated = curriculum_holder.get("validated", False)
        validation_result = curriculum_holder.get("forced_validation") or (
            "PASS" if is_validated else None
        )
        logger.info(
            "[agent] invoke done curriculum=%s validated=%s regen=%s",
            has_curriculum, is_validated, curriculum_holder.get("regen_count", 0),
        )
        return {
            "reply": reply,
            "complete": has_curriculum and is_validated,
            "curriculum": curriculum_holder.get("result"),
            "validation_result": validation_result,
        }
    except Exception as err:
        logger.error("[agent] error: %s", err)
        return {
            "reply": "죄송합니다. 처리 중 문제가 발생했습니다. 다시 시도해 주세요.",
            "complete": False,
            "curriculum": curriculum_holder.get("result"),
            "validation_result": None,
        }
