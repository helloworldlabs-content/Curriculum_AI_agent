import json
import os
from textwrap import dedent

from openai import OpenAI


# 현재 파일 위치를 기준으로 프로젝트 루트를 계산한다.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")


# 이 함수는 프로젝트 폴더에 있는 .env 파일을 읽어서
# OPENAI_API_KEY 같은 환경 변수를 파이썬 프로그램에서 사용할 수 있게 만든다.
# 보통은 python-dotenv 라이브러리를 쓰기도 하지만,
def load_env_file(env_path=ENV_PATH):
    # .env 파일이 없으면 그냥 넘어간다.
    if not os.path.exists(env_path):
        return

    # UTF-8로 파일을 읽으면 한글이 들어 있어도 비교적 안전하다.
    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            # 빈 줄, 주석 줄(#), = 없는 줄은 무시한다.
            if not line or line.startswith("#") or "=" not in line:
                continue

            # KEY=VALUE 형식으로 나누어 저장한다.
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # 이미 같은 이름의 환경 변수가 있으면 덮어쓰지 않는다.
            if key and key not in os.environ:
                os.environ[key] = value


# 콘솔에서 질문을 보여주고, 사용자의 답변을 문자열로 돌려준다.
def ask_question(prompt_text):
    return input(prompt_text).strip()


# 커리큘럼을 만들 때 필요한 최소 요구사항을 사용자에게 하나씩 입력받는다.
def collect_company_requirements():
    print("\n[기업 요구사항 입력]")
    print("질문에 답하면 OpenAI API로 교육 커리큘럼 초안을 생성합니다.\n")

    return {
        "company_name": ask_question("1. 회사명 또는 팀 이름: "),
        "goal": ask_question("2. 교육 목표: "),
        "audience": ask_question("3. 교육 대상자: "),
        "level": ask_question("4. 현재 AI 활용 수준 (예: 입문, 초급, 중급): "),
        "duration": ask_question("5. 교육 기간 또는 총 시간: "),
        "topic": ask_question("6. 원하는 핵심 주제: "),
        "constraints": ask_question("7. 꼭 반영해야 할 조건 또는 제한사항: "),
    }


# 사용자가 입력한 내용을 다시 보여주면,
# 어떤 정보가 모델에 전달되는지 눈으로 확인하기 쉽다.
def print_requirements_summary(requirements):
    print("\n[입력한 요구사항 요약]")
    print(f"- 회사/팀: {requirements['company_name']}")
    print(f"- 교육 목표: {requirements['goal']}")
    print(f"- 교육 대상자: {requirements['audience']}")
    print(f"- 현재 수준: {requirements['level']}")
    print(f"- 교육 기간: {requirements['duration']}")
    print(f"- 핵심 주제: {requirements['topic']}")
    print(f"- 제한사항: {requirements['constraints']}")


# system/developer 프롬프트는 모델의 역할과 출력 규칙을 정하는 부분이다.
# 여기서 "반드시 JSON으로만 답하라" 같은 규칙을 주면
# 파이썬 코드에서 결과를 다루기가 쉬워진다.
def build_system_prompt():
    return dedent(
        """
        당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
        사용자가 입력한 요구사항을 바탕으로 실무형 교육 커리큘럼을 설계하라.

        반드시 아래 규칙을 지켜라.
        1. 출력은 반드시 JSON 하나만 반환한다.
        2. 설명 문장, 코드블록 마크다운, 인사말을 추가하지 않는다.
        3. 회차는 4개 이상 6개 이하로 구성한다.
        4. 각 회차는 title, goals, activities를 포함한다.
        5. 마지막에는 expected_outcomes와 notes를 포함한다.
        6. notes에는 기업의 제한사항과 주의사항을 반영한다.
        """
    ).strip()


# 이 스키마는 "응답 JSON의 모양"을 미리 정해두는 역할을 한다.
# 이렇게 해두면 모델이 들쭉날쭉한 형식으로 답할 가능성이 줄어든다.
def build_curriculum_schema():
    return {
        "type": "json_schema",
        "name": "curriculum_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "program_title": {"type": "string"},
                "target_summary": {"type": "string"},
                "sessions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "goals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "activities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["title", "goals", "activities"],
                        "additionalProperties": False,
                    },
                },
                "expected_outcomes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "program_title",
                "target_summary",
                "sessions",
                "expected_outcomes",
                "notes",
            ],
            "additionalProperties": False,
        },
    }


# user 프롬프트에는 실제 기업 요구사항을 자연어로 넣는다.
# 즉, "무엇을 만들어야 하는지"를 모델에게 설명하는 부분이다.
def build_user_prompt(requirements):
    return dedent(
        f"""
        다음 기업 요구사항을 바탕으로 교육 커리큘럼 초안을 설계해줘.

        회사/팀: {requirements['company_name']}
        교육 목표: {requirements['goal']}
        교육 대상자: {requirements['audience']}
        현재 AI 활용 수준: {requirements['level']}
        교육 기간 또는 총 시간: {requirements['duration']}
        원하는 핵심 주제: {requirements['topic']}
        꼭 반영해야 할 조건 또는 제한사항: {requirements['constraints']}

        요구사항:
        - 초보자도 이해할 수 있도록 회차별 목표를 분명하게 작성
        - 실무 적용 가능성이 드러나도록 활동을 구성
        - 기업 교육이라는 점을 반영해서 너무 학문적으로 쓰지 말 것
        """
    ).strip()


# OpenAI Responses API 응답에서 실제 텍스트만 꺼내는 함수다.
# SDK 버전에 따라 output_text가 바로 있을 수도 있고,
# output 안쪽을 직접 순회해야 할 수도 있어서 둘 다 처리한다.
def extract_text_from_response(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    parts = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") == "output_text":
                parts.append(getattr(content, "text", ""))

    return "".join(parts).strip()


# 이 함수가 실제로 OpenAI API를 호출하는 핵심 부분이다.
# 1) API 키 확인
# 2) 모델 호출
# 3) JSON 문자열을 파이썬 딕셔너리로 변환
# 순서로 동작한다.
def generate_curriculum_with_openai(requirements):
    api_key = os.getenv("OPENAI_API_KEY")

    # API 키가 없으면 호출 자체가 불가능하므로 먼저 막아준다.
    if not api_key:
        raise ValueError(
            f"OPENAI_API_KEY를 찾을 수 없습니다. 다음 파일을 확인해주세요: {ENV_PATH}"
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-mini",
        text={"format": build_curriculum_schema()},
        input=[
            # developer 역할은 모델에게 지침을 주는 데 사용한다.
            {"role": "developer", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(requirements)},
        ],
    )

    raw_text = extract_text_from_response(response)
    if not raw_text:
        raise ValueError("모델 응답에서 텍스트를 읽지 못했습니다.")

    # 모델이 반환한 JSON 문자열을 파이썬 객체로 바꾼다.
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "모델 응답이 JSON 형식이 아니어서 해석할 수 없습니다.\n"
            f"응답 내용: {raw_text}"
        ) from error


# 생성된 커리큘럼을 사람이 읽기 편한 콘솔 형태로 출력한다.
def print_curriculum(curriculum):
    print("\n[추천 커리큘럼 초안]")
    print(f"\n과정명: {curriculum['program_title']}")
    print(f"대상 요약: {curriculum['target_summary']}")

    for index, session in enumerate(curriculum["sessions"], start=1):
        print(f"\n{index}. {session['title']}")

        print("  목표")
        for goal in session["goals"]:
            print(f"  - {goal}")

        print("  활동")
        for activity in session["activities"]:
            print(f"  - {activity}")

    print("\n예상 결과")
    for outcome in curriculum["expected_outcomes"]:
        print(f"- {outcome}")

    print("\n참고 사항")
    for note in curriculum["notes"]:
        print(f"- {note}")


# run_chatbot은 프로그램의 전체 흐름을 연결하는 함수다.
# 보통 이런 함수를 보면 "프로그램이 어떤 순서로 움직이는지"를 한눈에 파악할 수 있다.
def run_chatbot():
    # 가장 먼저 .env를 읽어서 API 키를 메모리에 올린다.
    load_env_file()

    print(
        dedent(
            """
            =========================================
            기업 교육용 AI 커리큘럼 설계 챗봇
            =========================================
            기업 요구사항을 입력하면 OpenAI API로 커리큘럼 초안을 생성합니다.
            """
        ).strip()
    )

    requirements = collect_company_requirements()
    print_requirements_summary(requirements)

    print("\n[생성 중]")
    print("입력한 요구사항을 바탕으로 커리큘럼 초안을 생성하고 있습니다...\n")

    curriculum = generate_curriculum_with_openai(requirements)
    print_curriculum(curriculum)


# 파이썬 파일을 직접 실행했을 때만 아래 코드가 동작한다.
# 다른 파일에서 import할 때는 자동 실행되지 않아서 재사용에 유리하다.
if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as error:
        print("\n[오류]")
        print(error)
