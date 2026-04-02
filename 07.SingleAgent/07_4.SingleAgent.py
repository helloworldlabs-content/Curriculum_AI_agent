import json
import logging
import os
from textwrap import dedent

from openai import OpenAI
from langchain_chroma import Chroma

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("single_agent.agent")

# ---------------------------------------------------------------------------
# 도구 정의
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, parameters: dict) -> dict:
    """OpenAI function tool 정의 헬퍼."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [
    _fn(
        "web_search",
        dedent("""
            Tavily API로 외부 웹을 검색한다.
            커리큘럼 생성 전 반드시 사용해야 하며, 다음 두 가지 목적으로 활용한다:
            1. 최근 AI 교육 트렌드 및 기업 도입 사례 파악
            2. 실제 강의 커리큘럼 구성 사례 (목차, 세션 구조, 실습 방식 등)
            대화 중 사용자가 언급한 기술이나 도구에 대한 최신 정보 확인에도 활용한다.
        """).strip(),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 쿼리 (한국어 또는 영어)"},
            },
            "required": ["query"],
        },
    ),
    _fn(
        "search_ax_compass",
        dedent("""
            내부 VectorDB에서 AX Compass 유형별 특성, 강점, 보완 방향, 교육 접근법을 검색한다.
            특정 유형(균형형, 이해형, 과신형, 실행형, 판단형, 조심형)에 대한 정보가 필요할 때 사용한다.
        """).strip(),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"},
                "type_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "필터링할 AX Compass 유형 목록 (예: ['균형형', '이해형']). 전체 검색은 빈 배열.",
                },
            },
            "required": ["query"],
        },
    ),
    _fn(
        "search_curriculum_examples",
        dedent("""
            내부 VectorDB에서 기업 AI 교육 커리큘럼 예시를 검색한다.
            사용자 요구사항(주제·수준·목표)과 일치하는 기존 커리큘럼이 있는지 확인할 때 사용한다.
            일치하는 예시가 있으면 generate_curriculum의 context에 포함시킨다.
            일치하는 예시가 없더라도 web_search로 외부 자료를 보완한다.
        """).strip(),
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
            수집된 정보와 에이전트가 직접 모은 참고 자료를 바탕으로 하루 단위 커리큘럼을 생성한다.
            커리큘럼은 1일차부터 N일차까지 하루씩 구성되며, 각 일차에는 공통 세션과 그룹별 세션이 포함된다.
            반드시 이 도구를 호출하기 전에 아래 검색을 모두 완료해야 한다:
              1. search_ax_compass × 3 — 그룹 A/B/C 유형별 특성 (내부 RAG)
              2. search_curriculum_examples × 1 이상 — 요구사항 일치 예시 확인 (내부 RAG)
              3. web_search × 2 이상 — 실제 커리큘럼 사례 + 최신 트렌드 (외부)
            각 검색 결과를 그룹별·용도별로 분리된 파라미터에 전달한다.
        """).strip(),
        {
            "type": "object",
            "properties": {
                "company_name":        {"type": "string",  "description": "회사명 또는 팀 이름"},
                "goal":                {"type": "string",  "description": "교육 목표"},
                "audience":            {"type": "string",  "description": "교육 대상자"},
                "level":               {"type": "string",  "description": "현재 AI 활용 수준 (입문/초급/중급)"},
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
                "ax_context_a": {
                    "type": "string",
                    "description": "search_ax_compass로 검색한 그룹 A(균형형·이해형) 유형 특성·강점·보완 방향",
                },
                "ax_context_b": {
                    "type": "string",
                    "description": "search_ax_compass로 검색한 그룹 B(과신형·실행형) 유형 특성·강점·보완 방향",
                },
                "ax_context_c": {
                    "type": "string",
                    "description": "search_ax_compass로 검색한 그룹 C(판단형·조심형) 유형 특성·강점·보완 방향",
                },
                "curriculum_context": {
                    "type": "string",
                    "description": "search_curriculum_examples로 검색한 내부 커리큘럼 예시 (없으면 빈 문자열)",
                },
                "web_context": {
                    "type": "string",
                    "description": "web_search로 수집한 실제 커리큘럼 사례 및 최신 트렌드",
                },
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
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼 설계를 위한 전문 에이전트다.
    사용자와 대화하며 필요한 정보를 수집하고, 스스로 참고 자료를 검색한 뒤 커리큘럼을 설계한다.

    ## 수집해야 할 정보 (9가지)
    1. 회사명 또는 팀 이름
    2. 교육 목표
    3. 교육 대상자
    4. 현재 AI 활용 수준 (입문 / 초급 / 중급)
    5. 총 교육 기간 (일수)
    6. 하루 교육 시간
    7. 다루고 싶은 주제
    8. 반영해야 할 조건 또는 제한사항
    9. AX Compass 진단 결과 — 6개 유형별 인원 (균형형·이해형·과신형·실행형·판단형·조심형)

    ## 커리큘럼 생성 전 필수 검색 절차
    9가지 정보가 모두 수집되면, generate_curriculum 호출 전에 반드시 아래 순서로 검색한다.

    [유형 정보 — 내부 RAG 필수 3회]
    1. search_ax_compass("균형형 이해형 특성 강점 교육 접근법", ["균형형", "이해형"])
    2. search_ax_compass("과신형 실행형 특성 강점 교육 접근법", ["과신형", "실행형"])
    3. search_ax_compass("판단형 조심형 특성 강점 교육 접근법", ["판단형", "조심형"])

    [커리큘럼 참고 자료 — 내부 RAG + 웹 검색 필수]
    4. search_curriculum_examples("사용자 주제·수준·목표 기반 쿼리")
       → 일치하는 예시가 있으면 context에 포함, 없어도 다음 단계로 진행
    5. web_search("{주제} 기업 교육 커리큘럼 사례")
       → 실제 강의 구성·세션 구조·실습 사례 수집
    6. web_search("{주제} 최신 트렌드 {연도}")
       → 최근 기술 동향 및 기업 도입 현황 파악

    위 검색 결과를 모두 하나의 문자열로 합쳐 generate_curriculum의 context 파라미터로 전달한다.

    ## 생성되는 커리큘럼 구조
    커리큘럼은 회차 단위가 아닌 **하루 단위**로 구성된다.
    - 교육 기간(days)만큼 일차별 스케줄(1일차, 2일차 ...)이 생성된다.
    - 각 일차에는 전체 공통 세션과 그룹별(A/B/C) 실습 세션이 포함된다.
    - 총 하루 시간 = 공통 세션 합계 + 그룹 세션 합계(한 그룹 기준)

    ## 도구별 사용 기준
    - search_ax_compass: 그룹별 유형 특성 (내부 RAG, 항상 필수)
    - search_curriculum_examples: 요구사항과 일치하는 기존 사례 확인 (내부 RAG, 항상 필수)
    - web_search: 실제 커리큘럼 사례 + 최신 트렌드 (외부, 커리큘럼 생성 전 항상 필수)
    - generate_curriculum: 위 모든 검색 완료 후 호출

    ## 주의사항
    - 사용자가 여러 정보를 한 번에 말하면 이미 채워진 항목은 다시 묻지 않는다.
    - AX Compass 인원은 6개 유형 모두 확인해야 한다 (0명도 명시 필요).
    - generate_curriculum은 반드시 위 1~6번 검색 이후에 호출한다.
    - 커리큘럼 생성 후 사용자가 수정을 요청하면 필요한 검색을 다시 수행하고 generate_curriculum을 재호출한다.
    """
).strip()


# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------

def _execute_tool(name: str, tool_input: dict, vectorstore: Chroma) -> tuple[str, dict | None]:
    """도구를 실행하고 (결과 문자열, 커리큘럼 dict 또는 None)을 반환한다."""
    import importlib.util, sys
    helpers_path = os.path.join(os.path.dirname(__file__), "07_3.AgentHelpers.py")
    if "agent_helpers" not in sys.modules:
        spec = importlib.util.spec_from_file_location("agent_helpers", helpers_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent_helpers"] = mod
        spec.loader.exec_module(mod)
    helpers = sys.modules["agent_helpers"]

    if name == "web_search":
        result = helpers.web_search(tool_input["query"])
        return result, None

    if name == "search_ax_compass":
        type_names = tool_input.get("type_names") or None
        result = helpers.retrieve_ax_compass(vectorstore, tool_input["query"], type_names)
        return result, None

    if name == "search_curriculum_examples":
        result = helpers.retrieve_curriculum_examples(vectorstore, tool_input["query"])
        return result, None

    if name == "generate_curriculum":
        ax_context_a      = tool_input.pop("ax_context_a", "")
        ax_context_b      = tool_input.pop("ax_context_b", "")
        ax_context_c      = tool_input.pop("ax_context_c", "")
        curriculum_context = tool_input.pop("curriculum_context", "")
        web_context       = tool_input.pop("web_context", "")
        curriculum = helpers.generate_curriculum(
            tool_input,
            ax_context_a=ax_context_a,
            ax_context_b=ax_context_b,
            ax_context_c=ax_context_c,
            curriculum_context=curriculum_context,
            web_context=web_context,
        )
        return "커리큘럼이 성공적으로 생성되었습니다.", curriculum

    return f"알 수 없는 도구: {name}", None


def run_agent(
    messages: list[dict],
    vectorstore: Chroma,
    *,
    max_iterations: int = 20,
) -> dict:
    """
    에이전트 루프를 실행한다.

    Parameters
    ----------
    messages : list[dict]
        클라이언트가 전송한 대화 기록 {"role": "user"|"assistant", "content": str}
    vectorstore : Chroma
        RAG에 사용할 벡터스토어
    max_iterations : int
        무한 루프 방지용 최대 반복 횟수

    Returns
    -------
    dict
        {
            "reply": str,       # 에이전트의 최종 텍스트 응답
            "complete": bool,   # 커리큘럼 생성 여부
            "curriculum": dict | None,
        }
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # system 메시지 포함 전체 대화를 OpenAI 형식으로 구성
    api_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    api_messages += [{"role": m["role"], "content": m["content"]} for m in messages]

    curriculum: dict | None = None

    for iteration in range(max_iterations):
        logger.info("[agent] iteration=%s messages=%s", iteration, len(api_messages))

        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message

        logger.info("[agent] finish_reason=%s", finish_reason)

        # 도구 호출이 없으면 최종 텍스트 응답으로 종료
        if finish_reason == "stop":
            return {"reply": message.content or "", "complete": curriculum is not None, "curriculum": curriculum}

        # 도구 호출 처리
        if finish_reason == "tool_calls":
            # assistant 메시지(tool_calls 포함) 추가
            api_messages.append(message)

            for tc in message.tool_calls:
                tool_args = json.loads(tc.function.arguments)
                logger.info("[agent] tool=%s input_keys=%s", tc.function.name, list(tool_args.keys()))

                result_text, maybe_curriculum = _execute_tool(tc.function.name, tool_args, vectorstore)

                if maybe_curriculum is not None:
                    curriculum = maybe_curriculum

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
            continue

        # 예상치 못한 finish_reason
        logger.warning("[agent] unexpected finish_reason=%s", finish_reason)
        break

    # max_iterations 초과
    logger.error("[agent] max_iterations(%s) exceeded", max_iterations)
    return {
        "reply": "죄송합니다. 처리 중 문제가 발생했습니다. 다시 시도해 주세요.",
        "complete": curriculum is not None,
        "curriculum": curriculum,
    }
