import os
from textwrap import dedent
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from schemas import CollectedInfo, CurriculumPlan, Message


# 이 파일은 "무엇을 찾을지"와 "찾은 내용을 어떻게 LLM에 넘길지"를 담당한다.
COLLECTION_SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼 설계를 위한 정보 수집 어시스턴트다.
    아래 항목을 자연스러운 대화로 한 번에 1개씩 순서대로 수집해라.
    사용자가 여러 정보를 한 번에 말하면 파악하고 빠진 항목만 추가 질문해라.
    모든 항목이 수집되면 수집한 정보를 요약하고 마지막에 반드시 "[정보 수집 완료]"를 출력해라.

    수집 항목:
    - 회사명 또는 팀 이름
    - 교육 목표
    - 교육 대상자
    - 현재 AI 활용 수준 (입문/초급/중급)
    - 총 교육 기간 (일수, 숫자)
    - 하루 교육 시간 (시간, 숫자)
    - 원하는 핵심 주제
    - 반영해야 할 조건 또는 제한사항
    - AX Compass 진단 결과: 6개 유형별 인원수 (균형형, 이해형, 과신형, 실행형, 판단형, 조심형)
    """
).strip()

GENERATION_SYSTEM_PROMPT = dedent(
    """
    당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
    앞선 대화에서 수집한 기업 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계하라.

    커리큘럼 구조:
    - theory_sessions: 모든 참가자가 동일하게 수강하는 공통 이론 수업. 4개 이상 6개 이하.
    - group_sessions: 3개 그룹이 각각 다른 실습을 진행. 각 그룹당 2개 이상 3개 이하.

    규칙:
    1. 그룹별 실습은 해당 유형의 강점을 활용하고 보완 방향을 실습 활동에 녹인다.
    2. 각 회차는 title, duration_hours, goals, activities를 포함한다.
    3. duration_hours는 시간 단위(소수점 가능)이며, 아래 시간 배분 규칙을 따른다:
       - theory_sessions는 전체 참가자가 순차적으로 수강하므로 duration_hours 합산이 전체 시간에 포함된다.
       - group_sessions는 3개 그룹이 동시에 진행되므로, 각 그룹의 duration_hours 합산은 모두 동일해야 한다.
       - theory_sessions 합계 + 그룹 실습 합계(1개 그룹 기준) = 총 교육 시간
    4. 기업 교육답게 실무 적용 중심으로 구성한다.
    5. notes에는 유형별 특성과 기업 제한사항을 반영한 주의사항을 작성한다.
    6. [커리큘럼 예시 참고 자료]가 제공되는 경우, 아래 기준으로 참고하되 내용을 그대로 복사하지 마라:
       - 세션 제목과 구성 방식 (이론->실습 흐름, 주제 전개 순서 등)
       - 활동 유형 (워크숍, 실습, 토론, 케이스 스터디 등)
       - 강의 내용의 깊이와 수준 (용어, 난이도, 현업 적용 방식)
    """
).strip()

_MSG_CLS = {"user": HumanMessage, "assistant": AIMessage}


def to_lc_messages(messages: list[Message]) -> list:
    # FastAPI 요청 형식을 LangChain 메시지 형식으로 바꿔, LLM에 바로 전달할 수 있게 한다.
    return [_MSG_CLS[m.role](content=m.content) for m in messages]


def _retrieve(vectorstore: Chroma, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Document]:
    # 검색 공통 함수. 필요하면 메타데이터 필터까지 함께 건다.
    search_kwargs: dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs).invoke(query)


def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    # AX Compass 쪽은 "특정 유형"만 보고 싶으므로 type_name 필터를 함께 사용한다.
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    docs = _retrieve(
        vectorstore,
        query,
        max(4, len(type_names) * 3),
        {"type_name": {"$in": type_names}},
    )
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_curriculum_examples(vectorstore: Chroma, query: str, k: int = 3) -> str:
    # 커리큘럼 예시는 유사한 사례를 몇 개 가져와 생성 단계의 참고 자료로 쓴다.
    docs = _retrieve(vectorstore, query, k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstores: dict[str, Chroma]):
    # 최종 체인은 "검색 -> 프롬프트 조립 -> 구조화 출력" 순서로 동작한다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        # 사용자가 준 정보와 검색 결과를 한 번에 모아, 커리큘럼 생성용 최종 메시지를 만든다.
        conversation = input_dict["conversation"]
        info: CollectedInfo = input_dict["info"]
        groups = input_dict["groups"]

        # 시간 계산을 먼저 고정해 두면, LLM이 회차별 시간을 맞출 때 흔들림이 줄어든다.
        total_hours = info.days * info.hours_per_day
        theory_hours = round(total_hours * 0.65)
        group_hours = total_hours - theory_hours

        ax_vectorstore = vectorstores["ax_compass"]
        curriculum_vectorstore = vectorstores["curriculum_examples"]

        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]

        # 각 그룹이 참고해야 할 AX Compass 특성을 따로 가져온다.
        ctx_a = retrieve_group_context(ax_vectorstore, ga["types"])
        ctx_b = retrieve_group_context(ax_vectorstore, gb["types"])
        ctx_c = retrieve_group_context(ax_vectorstore, gc["types"])

        # 전체 프로그램 분위기와 유사한 커리큘럼 예시도 함께 붙인다.
        curriculum_query = f"{info.topic} {info.level} 기업 AI 교육 커리큘럼"
        curriculum_examples = retrieve_curriculum_examples(curriculum_vectorstore, curriculum_query)

        # 새 시스템 프롬프트를 쓸 것이므로, 기존 대화의 SystemMessage는 제외한다.
        chat_history = [message for message in conversation if not isinstance(message, SystemMessage)]

        # 검색 결과를 사람이 읽는 참고자료처럼 정리해서 마지막 HumanMessage에 넣는다.
        rag_content = dedent(
            f"""
            위 대화에서 수집한 요구사항을 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

            [시간 배분 기준]
            총 교육 시간: {total_hours}시간
            - 이론 수업(theory_sessions) duration_hours 합계: 정확히 {theory_hours}시간
            - 그룹 실습(group_sessions) duration_hours 합계 (1개 그룹 기준): 정확히 {group_hours}시간
            - group_sessions는 3개 그룹이 동시에 진행되므로 각 그룹의 duration_hours 합산은 모두 동일해야 한다.

            [그룹 구성]
            - {ga['name']} ({' · '.join(ga['types'])}): {ga['count']}명
            - {gb['name']} ({' · '.join(gb['types'])}): {gb['count']}명
            - {gc['name']} ({' · '.join(gc['types'])}): {gc['count']}명

            [AX Compass 유형별 특성 — 벡터 DB 검색 결과]
            === 그룹 A ({' · '.join(ga['types'])}) ===
            {ctx_a}

            === 그룹 B ({' · '.join(gb['types'])}) ===
            {ctx_b}

            === 그룹 C ({' · '.join(gc['types'])}) ===
            {ctx_c}

            [커리큘럼 예시 참고 자료 — 세션 구성 방식·활동 유형·강의 내용 수준을 참고할 것]
            {curriculum_examples}
            """
        ).strip()

        return [SystemMessage(content=GENERATION_SYSTEM_PROMPT)] + chat_history + [HumanMessage(content=rag_content)]

    return RunnableLambda(retrieve_and_build_messages) | structured_llm
