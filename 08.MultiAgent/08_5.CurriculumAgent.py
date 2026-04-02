import json
import logging
import os
from textwrap import dedent

from openai import OpenAI
from langchain_chroma import Chroma

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.curriculum_agent")

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼을 생성하는 전문 에이전트다.
    정보 수집 에이전트로부터 전달받은 요구사항을 바탕으로 커리큘럼을 설계한다.

    ## 커리큘럼 생성 절차 (반드시 순서대로)
    1. search_ax_compass("균형형 이해형 특성 강점 교육 접근법", ["균형형", "이해형"])
    2. search_ax_compass("과신형 실행형 특성 강점 교육 접근법", ["과신형", "실행형"])
    3. search_ax_compass("판단형 조심형 특성 강점 교육 접근법", ["판단형", "조심형"])
    4. search_curriculum_examples(사용자 주제·수준·목표 기반 쿼리)
    5. web_search("{주제} 기업 교육 커리큘럼 사례")
    6. web_search("{주제} 최신 트렌드")
    7. generate_curriculum (위 검색 완료 후에만 호출)

    ## 주의사항
    - 평가 에이전트의 개선 요청이 있으면 해당 내용을 generate_curriculum 호출 시 반드시 반영한다.
    - generate_curriculum은 위 1~6번 검색을 모두 완료한 뒤에만 호출한다.
    - 검색 결과를 그룹별·용도별로 분리해서 해당 파라미터에 전달한다.
    """
).strip()

# ---------------------------------------------------------------------------
# 도구 정의
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [
    _fn(
        "web_search",
        "Tavily API로 외부 웹을 검색한다. 최신 AI 교육 트렌드 및 실제 커리큘럼 사례를 수집할 때 사용한다.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"},
            },
            "required": ["query"],
        },
    ),
    _fn(
        "search_ax_compass",
        "내부 VectorDB에서 AX Compass 유형별 특성, 강점, 보완 방향, 교육 접근법을 검색한다.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"},
                "type_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "필터링할 AX Compass 유형 목록 (예: ['균형형', '이해형'])",
                },
            },
            "required": ["query"],
        },
    ),
    _fn(
        "search_curriculum_examples",
        "내부 VectorDB에서 기업 AI 교육 커리큘럼 예시를 검색한다.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리 (주제, 수준, 목표 포함)"},
            },
            "required": ["query"],
        },
    ),
    _fn(
        "generate_curriculum",
        dedent("""
            수집된 정보와 검색한 참고 자료를 바탕으로 하루 단위 커리큘럼을 생성한다.
            반드시 search_ax_compass × 3, search_curriculum_examples × 1, web_search × 2 완료 후 호출한다.
        """).strip(),
        {
            "type": "object",
            "properties": {
                "company_name":        {"type": "string",  "description": "회사명 또는 팀 이름"},
                "goal":                {"type": "string",  "description": "교육 목표"},
                "audience":            {"type": "string",  "description": "교육 대상자"},
                "level":               {"type": "string",  "description": "현재 AI 활용 수준"},
                "days":                {"type": "integer", "description": "총 교육 기간 (일수)"},
                "hours_per_day":       {"type": "integer", "description": "하루 교육 시간 (시간)"},
                "topic":               {"type": "string",  "description": "원하는 핵심 주제"},
                "constraints":         {"type": "string",  "description": "반영해야 할 조건 또는 제한사항"},
                "count_balanced":      {"type": "integer", "description": "균형형 인원수"},
                "count_learner":       {"type": "integer", "description": "이해형 인원수"},
                "count_overconfident": {"type": "integer", "description": "과신형 인원수"},
                "count_doer":          {"type": "integer", "description": "실행형 인원수"},
                "count_analyst":       {"type": "integer", "description": "판단형 인원수"},
                "count_cautious":      {"type": "integer", "description": "조심형 인원수"},
                "ax_context_a": {"type": "string", "description": "그룹 A(균형형·이해형) AX Compass 검색 결과"},
                "ax_context_b": {"type": "string", "description": "그룹 B(과신형·실행형) AX Compass 검색 결과"},
                "ax_context_c": {"type": "string", "description": "그룹 C(판단형·조심형) AX Compass 검색 결과"},
                "curriculum_context": {"type": "string", "description": "내부 커리큘럼 예시 검색 결과"},
                "web_context": {"type": "string", "description": "웹 검색으로 수집한 사례 및 트렌드"},
            },
            "required": [
                "company_name", "goal", "audience", "level",
                "days", "hours_per_day", "topic", "constraints",
                "count_balanced", "count_learner", "count_overconfident",
                "count_doer", "count_analyst", "count_cautious",
                "ax_context_a", "ax_context_b", "ax_context_c",
                "curriculum_context", "web_context",
            ],
        },
    ),
]

# ---------------------------------------------------------------------------
# 도구 실행
# ---------------------------------------------------------------------------

def _execute_tool(name: str, tool_input: dict, vectorstore: Chroma) -> tuple[str, dict | None]:
    import importlib.util, sys
    helpers_path = os.path.join(os.path.dirname(__file__), "08_3.AgentHelpers.py")
    if "agent_helpers_08" not in sys.modules:
        spec = importlib.util.spec_from_file_location("agent_helpers_08", helpers_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent_helpers_08"] = mod
        spec.loader.exec_module(mod)
    helpers = sys.modules["agent_helpers_08"]

    if name == "web_search":
        return helpers.web_search(tool_input["query"]), None

    if name == "search_ax_compass":
        type_names = tool_input.get("type_names") or None
        return helpers.retrieve_ax_compass(vectorstore, tool_input["query"], type_names), None

    if name == "search_curriculum_examples":
        return helpers.retrieve_curriculum_examples(vectorstore, tool_input["query"]), None

    if name == "generate_curriculum":
        ax_context_a       = tool_input.pop("ax_context_a", "")
        ax_context_b       = tool_input.pop("ax_context_b", "")
        ax_context_c       = tool_input.pop("ax_context_c", "")
        curriculum_context = tool_input.pop("curriculum_context", "")
        web_context        = tool_input.pop("web_context", "")
        # evaluator_feedback는 에이전트가 시스템 프롬프트로 이미 알고 있으므로
        # helpers.generate_curriculum에 별도 전달
        evaluator_feedback = tool_input.pop("evaluator_feedback", "")
        curriculum = helpers.generate_curriculum(
            tool_input,
            ax_context_a=ax_context_a,
            ax_context_b=ax_context_b,
            ax_context_c=ax_context_c,
            curriculum_context=curriculum_context,
            web_context=web_context,
            evaluator_feedback=evaluator_feedback,
        )
        return "커리큘럼이 성공적으로 생성되었습니다.", curriculum

    return f"알 수 없는 도구: {name}", None

# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------

def run_curriculum_agent(
    collected_info: dict,
    vectorstore: Chroma,
    *,
    evaluator_feedback: str = "",
    regen_count: int = 0,
    max_iterations: int = 20,
) -> dict:
    """
    수집된 정보를 받아 RAG 검색 후 커리큘럼을 생성한다.

    Parameters
    ----------
    collected_info : dict
        InfoCollectorAgent가 수집한 9가지 정보
    vectorstore : Chroma
        RAG에 사용할 벡터스토어
    evaluator_feedback : str
        EvaluatorAgent가 요청한 개선사항 (재생성 시에만 전달)
    regen_count : int
        재생성 횟수 (로깅 용도)
    max_iterations : int
        무한 루프 방지용 최대 반복 횟수

    Returns
    -------
    dict
        {
            "reply": str,
            "curriculum": dict | None,
        }
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system_content = SYSTEM_PROMPT
    if evaluator_feedback:
        system_content += (
            f"\n\n## 평가 에이전트의 개선 요청 (재생성 #{regen_count})\n"
            f"{evaluator_feedback}\n"
            "위 개선 요청을 generate_curriculum 호출 시 반드시 반영해야 한다."
        )

    # 에이전트에게 수집된 정보를 컨텍스트로 제공
    user_message = (
        f"아래 정보를 바탕으로 커리큘럼을 생성해 주세요.\n\n"
        f"```json\n{json.dumps(collected_info, ensure_ascii=False, indent=2)}\n```"
    )
    if evaluator_feedback:
        user_message += f"\n\n평가 에이전트의 피드백:\n{evaluator_feedback}"

    api_messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_message},
    ]

    curriculum: dict | None = None

    for iteration in range(max_iterations):
        logger.info("[curriculum_agent] iteration=%s regen=%s", iteration, regen_count)

        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "stop":
            return {
                "reply": message.content or "",
                "curriculum": curriculum,
            }

        if choice.finish_reason == "tool_calls":
            api_messages.append(message)

            for tc in message.tool_calls:
                tool_args = json.loads(tc.function.arguments)
                logger.info("[curriculum_agent] tool=%s", tc.function.name)

                result_text, maybe_curriculum = _execute_tool(tc.function.name, tool_args, vectorstore)
                if maybe_curriculum is not None:
                    curriculum = maybe_curriculum

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
            continue

        logger.warning("[curriculum_agent] unexpected finish_reason=%s", choice.finish_reason)
        break

    logger.error("[curriculum_agent] max_iterations exceeded")
    return {
        "reply": "커리큘럼 생성 중 문제가 발생했습니다. 다시 시도해 주세요.",
        "curriculum": curriculum,
    }
