import json
import logging
import os
from textwrap import dedent

from openai import OpenAI

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.info_collector")

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼 설계를 위한 정보 수집 전담 에이전트다.
    사용자와 친절하고 자연스럽게 대화하며 아래 9가지 정보를 빠짐없이 수집한다.

    ## 수집해야 할 정보
    1. 회사명 또는 팀 이름
    2. 교육 목표
    3. 교육 대상자
    4. 현재 AI 활용 수준 (입문 / 초급 / 중급)
    5. 총 교육 기간 (일수)
    6. 하루 교육 시간 (시간)
    7. 다루고 싶은 핵심 주제
    8. 반영해야 할 조건 또는 제한사항 (없으면 "없음"으로 처리)
    9. AX Compass 진단 결과 — 6개 유형별 인원수
       균형형 / 이해형 / 과신형 / 실행형 / 판단형 / 조심형
       (0명인 유형도 반드시 확인해야 함)

    ## 행동 지침
    - 사용자가 여러 정보를 한 번에 말하면 이미 확인된 항목은 다시 묻지 않는다.
    - 아직 수집되지 않은 정보를 자연스럽게 이어서 질문한다.
    - AX Compass 인원은 6개 유형 모두 확인되어야 한다 (0명도 명시).
    - 모든 정보가 수집되면 사용자에게 수집된 내용을 간략히 요약해 확인을 받은 뒤
      즉시 submit_info 도구를 호출한다.
    - 커리큘럼 생성은 이 에이전트의 역할이 아니다. 정보 수집에만 집중한다.
    """
).strip()

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

# ---------------------------------------------------------------------------
# 도구 정의
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_info",
            "description": (
                "9가지 정보가 모두 수집되고 사용자 확인까지 완료됐을 때 호출한다. "
                "수집된 정보를 구조화해서 커리큘럼 생성 에이전트에게 넘긴다."
            ),
            "parameters": {
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
                },
                "required": [
                    "company_name", "goal", "audience", "level",
                    "days", "hours_per_day", "topic", "constraints",
                    "count_balanced", "count_learner", "count_overconfident",
                    "count_doer", "count_analyst", "count_cautious",
                ],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------

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
        유사 커리큘럼 감지 시 오케스트레이터가 전달하는 컨텍스트.
        비어 있지 않으면 추가 정보 수집 모드로 동작한다.
    max_iterations : int
        무한 루프 방지용 최대 반복 횟수

    Returns
    -------
    dict
        {
            "reply": str,
            "collected_info": dict | None,
            "complete": bool,
        }
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system_content = SYSTEM_PROMPT

    # 유사 커리큘럼 감지 시: 추가 정보 수집 모드 프롬프트 삽입
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

    api_messages: list[dict] = [{"role": "system", "content": system_content}]
    api_messages += messages

    for iteration in range(max_iterations):
        logger.info("[info_collector] iteration=%s messages=%s", iteration, len(api_messages))

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
                "collected_info": None,
                "complete": False,
            }

        if choice.finish_reason == "tool_calls":
            api_messages.append(message)
            collected_info = None

            for tc in message.tool_calls:
                if tc.function.name == "submit_info":
                    collected_info = json.loads(tc.function.arguments)
                    logger.info("[info_collector] submit_info called: %s", list(collected_info.keys()))
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "정보가 성공적으로 제출되었습니다. 커리큘럼 생성 에이전트에게 전달합니다.",
                    })

            if collected_info is not None:
                # 제출 후 사용자에게 보낼 마지막 메시지를 생성
                final_response = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    tools=TOOLS,
                    tool_choice="none",
                )
                reply = final_response.choices[0].message.content or ""
                return {
                    "reply": reply,
                    "collected_info": collected_info,
                    "complete": True,
                }
            continue

        logger.warning("[info_collector] unexpected finish_reason=%s", choice.finish_reason)
        break

    logger.error("[info_collector] max_iterations exceeded")
    return {
        "reply": "정보 수집 중 문제가 발생했습니다. 다시 시도해 주세요.",
        "collected_info": None,
        "complete": False,
    }
