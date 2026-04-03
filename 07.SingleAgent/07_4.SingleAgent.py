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


def _build_tools(vectorstore: Chroma, curriculum_holder: dict) -> list:
    """
    vectorstore와 curriculum_holder를 클로저로 캡처한 LangChain 도구 목록을 반환한다.
    요청마다 새 그래프를 만들 때 호출하므로 매번 새 도구 객체가 생성된다.
    """
    helpers = _load_helpers()

    @tool
    def web_search(query: str) -> str:
        """
        Tavily API로 웹을 검색한다.
        커리큘럼 생성 시 반드시 사용해야 하며, 다음 두 가지 목적에 사용한다.
        1. 최근 AI 교육 트렌드 및 기업 도입 사례 파악
        2. 실제 강의 커리큘럼 구성 사례 확인
        """
        return helpers.web_search(query)

    @tool
    def search_ax_compass(query: str, type_names: list[str] | None = None) -> str:
        """
        내부 VectorDB에서 AX Compass 유형별 특성, 강점, 보완 방향, 교육 접근법을 검색한다.
        특정 유형에 대한 정보가 필요할 때 사용한다.
        type_names는 필터링할 AX Compass 유형 목록이다.
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
        반드시 search_ax_compass x 3, search_curriculum_examples x 1, web_search x 2 완료 후 호출한다.

        ax_context_a: 그룹 A(균형형·이해형) AX Compass 검색 결과
        ax_context_b: 그룹 B(과신형·실행형) AX Compass 검색 결과
        ax_context_c: 그룹 C(판단형·조심형) AX Compass 검색 결과
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
        curriculum_holder["result"] = curriculum
        return "커리큘럼이 성공적으로 생성되었습니다."

    return [web_search, search_ax_compass, search_curriculum_examples, generate_curriculum]


def run_agent(
    messages: list[dict],
    vectorstore: Chroma,
    *,
    max_iterations: int = 20,
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
        recursion_limit = max_iterations * 2

    Returns
    -------
    dict
        {"reply": str, "complete": bool, "curriculum": dict | None}
    """
    curriculum_holder: dict = {}

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tools = _build_tools(vectorstore, curriculum_holder)

    graph = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)

    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]

    logger.info("[agent] invoke start messages=%s", len(lc_messages))

    try:
        result = graph.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": max_iterations * 2},
        )
        last = result["messages"][-1]
        reply = last.content if isinstance(last.content, str) else str(last.content)
        logger.info("[agent] invoke done curriculum=%s", "result" in curriculum_holder)
        return {
            "reply": reply,
            "complete": "result" in curriculum_holder,
            "curriculum": curriculum_holder.get("result"),
        }
    except Exception as err:
        logger.error("[agent] error: %s", err)
        return {
            "reply": "죄송합니다. 처리 중 문제가 발생했습니다. 다시 시도해 주세요.",
            "complete": "result" in curriculum_holder,
            "curriculum": curriculum_holder.get("result"),
        }
