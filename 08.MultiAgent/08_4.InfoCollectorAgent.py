import json
import logging
import os
from pathlib import Path
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.info_collector")

PROMPT_PATH = Path(__file__).with_name("08_12.InfoCollectorPrompt.txt")


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


BASE_SYSTEM_PROMPT = _load_system_prompt()

# 유사 커리큘럼 감지 시 추가되는 시스템 프롬프트 섹션
_SIMILARITY_MODE_PROMPT = dedent(
    """
    ## 추가 정보 수집 모드 (커리큘럼 품질 개선)
    {similarity_context}

    커리큘럼 생성 에이전트가 동일한 방향의 커리큘럼을 반복 생성하고 있습니다.
    이를 개선하려면 사용자에게 새로운 방향성을 제시할 추가 정보가 필요합니다.

    ## 이 모드에서의 행동 지침
    1. 먼저 상황을 간략히 설명한다:
       "커리큘럼 재생성을 시도했지만 비슷한 구성이 반복되고 있어, 방향을 다시 잡기 위해
       몇 가지 추가 정보를 여쭤보겠습니다."

    2. 아래 항목 중 기존 정보로 파악하기 어려운 부분을 집중적으로 질문한다:
       - 현재 커리큘럼에서 특별히 바꾸고 싶은 점 (세션 방식, 주제 깊이, 그룹 구성 등)
       - 반드시 포함해야 하는 특정 주제나 활동 (구체적인 도구·사례·실습 방법 등)
       - 반드시 제외해야 하는 내용
       - 교육 대상자의 업무 특성이나 현재 고충 (더 구체적인 맥락)
       - 과거에 진행한 교육이 있다면 잘 됐거나 안 됐던 이유

    3. 사용자의 답변을 반영해 기존 정보(이미 수집된 정보)를 업데이트한다.
       - 변경이 없는 항목은 기존 값을 그대로 유지한다.
       - 사용자가 새로 제공한 내용은 goal, topic, constraints 등 적절한 항목에 반영한다.

    4. 추가 정보 수집 후 submit_info를 호출할 때 기존 항목과 새 내용을 합산해 전달한다.
    """
).strip()


def _build_tools(collected_info_holder: dict) -> list:
    @tool
    def submit_info(
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
    ) -> str:
        """
        9가지 정보가 모두 수집되고 사용자 확인까지 완료됐을 때 호출한다.
        수집된 정보를 구조화해서 커리큘럼 생성 에이전트에게 넘긴다.
        """
        collected_info_holder["result"] = {
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
        logger.info("[info_collector] submit_info called: %s", list(collected_info_holder["result"].keys()))
        return "정보가 성공적으로 제출되었습니다. 커리큘럼 생성 에이전트에게 전달합니다."

    return [submit_info]


def run_info_collector(
    messages: list[dict],
    existing_info: dict | None = None,
    *,
    similarity_context: str = "",
    max_iterations: int = 10,
) -> dict:
    """
    사용자 메시지를 받아 9가지 정보를 수집한다.

    Parameters
    ----------
    messages : list[dict]
        전체 대화 기록 {"role": "user"|"assistant", "content": str}
    existing_info : dict | None
        이미 수집된 정보 (부분 수집 재시작 시 전달)
    similarity_context : str
        유사 커리큘럼 감지 시 오케스트레이터가 전달하는 컨텍스트
    max_iterations : int
        recursion_limit = max_iterations * 2

    Returns
    -------
    dict
        {
            "reply": str,
            "collected_info": dict | None,
            "complete": bool,
        }
    """
    collected_info_holder: dict = {}

    system_content = BASE_SYSTEM_PROMPT

    if similarity_context:
        system_content += "\n\n" + _SIMILARITY_MODE_PROMPT.format(
            similarity_context=similarity_context
        )
        logger.info("[info_collector] similarity mode activated")

    if existing_info:
        system_content += (
            f"\n\n## 이미 수집된 정보\n"
            f"{json.dumps(existing_info, ensure_ascii=False, indent=2)}\n"
            "위 항목은 이미 확인된 정보이므로 변경 요청이 없는 한 그대로 유지한다."
        )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tools = _build_tools(collected_info_holder)
    graph = create_react_agent(model, tools, prompt=system_content)

    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]

    logger.info("[info_collector] invoke start messages=%s", len(lc_messages))

    try:
        result = graph.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": max_iterations * 2},
        )
        last = result["messages"][-1]
        reply = last.content if isinstance(last.content, str) else str(last.content)

        if "result" in collected_info_holder:
            logger.info("[info_collector] complete=True")
            return {
                "reply": reply,
                "collected_info": collected_info_holder["result"],
                "complete": True,
            }

        logger.info("[info_collector] complete=False")
        return {
            "reply": reply,
            "collected_info": None,
            "complete": False,
        }
    except Exception as err:
        logger.error("[info_collector] error: %s", err)
        return {
            "reply": "정보 수집 중 문제가 발생했습니다. 다시 시도해 주세요.",
            "collected_info": None,
            "complete": False,
        }
