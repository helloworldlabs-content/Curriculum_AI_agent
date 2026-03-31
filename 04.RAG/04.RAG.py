import os
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# ─── 경로 설정 ────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH        = os.path.join(BASE_DIR, ".env")
DATA_DIR        = os.path.join(BASE_DIR, "Data")
PDF_PATH        = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_v2"

TYPE_MARKERS = {
    "균형형": "## 1) 균형형",
    "실행형": "## 2) 실행형",
    "판단형": "## 3) 판단형",
    "이해형": "## 4) 이해형",
    "과신형": "## 5) 과신형",
    "조심형": "## 6) 조심형",
}

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


# ─── Pydantic 스키마 ──────────────────────────────────────────────────────────

class Session(BaseModel):
    title: str
    duration_hours: float
    goals: list[str]
    activities: list[str]


class GroupSession(BaseModel):
    group_name: str
    target_types: str
    participant_count: int
    focus_description: str
    sessions: list[Session]


class CurriculumPlan(BaseModel):
    program_title: str
    target_summary: str
    theory_sessions: list[Session]
    group_sessions: list[GroupSession]
    expected_outcomes: list[str]
    notes: list[str]


class CollectedInfo(BaseModel):
    company_name:      str = Field(description="회사명 또는 팀 이름")
    goal:              str = Field(description="교육 목표")
    audience:          str = Field(description="교육 대상자")
    level:             str = Field(description="현재 AI 활용 수준")
    days:              int = Field(description="총 교육 기간 (일수)")
    hours_per_day:     int = Field(description="하루 교육 시간 (시간)")
    topic:             str = Field(description="원하는 핵심 주제")
    constraints:       str = Field(description="반영해야 할 조건 또는 제한사항")
    count_balanced:    int = Field(description="균형형 인원수")
    count_learner:     int = Field(description="이해형 인원수")
    count_overconfident: int = Field(description="과신형 인원수")
    count_doer:        int = Field(description="실행형 인원수")
    count_analyst:     int = Field(description="판단형 인원수")
    count_cautious:    int = Field(description="조심형 인원수")


# ─── 환경 설정 ────────────────────────────────────────────────────────────────

def load_env_file():
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ─── RAG 1단계: PDF 로드 및 청크 분할 ───────────────────────────────────────

def load_and_split_documents() -> list:
    all_docs = []

    # AX Compass PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    print("[RAG] AXCompass PDF 로드 중...")
    ax_pages = PyPDFLoader(PDF_PATH).load()
    for page in ax_pages:
        page.metadata["doc_type"] = "ax_compass"
    all_docs.extend(ax_pages)

    # 커리큘럼 예시 PDF (AXCompass.pdf 제외)
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".pdf") and fname != "AXCompass.pdf":
            fpath = os.path.join(DATA_DIR, fname)
            print(f"[RAG] 커리큘럼 PDF 로드: {fname}")
            course_name = os.path.splitext(fname)[0]
            pages = PyPDFLoader(fpath).load()
            for page in pages:
                page.metadata.update({"doc_type": "curriculum_example", "course_name": course_name})
            all_docs.extend(pages)

    # 커리큘럼 예시 Excel
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".xlsx"):
            fpath = os.path.join(DATA_DIR, fname)
            print(f"[RAG] 커리큘럼 Excel 로드: {fname}")
            course_name = os.path.splitext(fname)[0]
            docs = UnstructuredExcelLoader(fpath).load()
            for doc in docs:
                doc.metadata.update({"doc_type": "curriculum_example", "course_name": course_name})
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)

    for chunk in chunks:
        if chunk.metadata.get("doc_type") == "ax_compass":
            for type_name, info in TYPE_INFO.items():
                if type_name in chunk.page_content:
                    chunk.metadata.update({
                        "type_name": type_name,
                        "group":     info["group"],
                        "english":   info["english"],
                    })
                    break

    ax_count = sum(1 for c in chunks if c.metadata.get("doc_type") == "ax_compass")
    ex_count = len(chunks) - ax_count
    print(f"[RAG] 총 {len(chunks)}개 청크 (AX Compass: {ax_count}, 커리큘럼 예시: {ex_count})")
    return chunks


# ─── RAG 2단계: 벡터 DB 구축 ─────────────────────────────────────────────────

def setup_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    if vectorstore._collection.count() == 0:
        print("[VectorDB] 임베딩 생성 중...")
        chunks = load_and_split_documents()
        vectorstore.add_documents(chunks)
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료 → {VECTOR_DB_PATH}")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")

    return vectorstore


# ─── RAG 3단계: 시맨틱 검색 ──────────────────────────────────────────────────

def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": len(type_names),
            "filter": {"$and": [
                {"doc_type":  {"$eq": "ax_compass"}},
                {"type_name": {"$in": type_names}},
            ]},
        },
    )
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


def retrieve_curriculum_examples(vectorstore: Chroma, query: str, k: int = 3) -> str:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": {"$eq": "curriculum_example"}},
        },
    )
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ─── RAG 4단계: LCEL 체인 구성 ───────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = dedent("""
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
       - 세션 제목과 구성 방식 (이론→실습 흐름, 주제 전개 순서 등)
       - 활동 유형 (워크숍, 실습, 토론, 케이스 스터디 등)
       - 강의 내용의 깊이와 수준 (용어, 난이도, 현업 적용 방식)
""").strip()


def build_chain(vectorstore: Chroma):
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation  = input_dict["conversation"]
        groups        = input_dict["groups"]
        total_hours   = input_dict["total_hours"]
        topic         = input_dict.get("topic", "")
        level         = input_dict.get("level", "")
        ga, gb, gc    = groups["group_a"], groups["group_b"], groups["group_c"]

        # ① Retrieve
        print("[RAG] 벡터 DB에서 유형별 컨텍스트를 검색하는 중...")
        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        print("[RAG] 커리큘럼 예시를 검색하는 중...")
        curriculum_query = f"{topic} {level} 기업 AI 교육 커리큘럼"
        curriculum_examples = retrieve_curriculum_examples(vectorstore, curriculum_query)

        # ② 대화 히스토리에서 SystemMessage 제외
        chat_history = [m for m in conversation if not isinstance(m, SystemMessage)]

        # ③ RAG 컨텍스트 + 생성 요청 메시지
        theory_hours = round(total_hours * 0.65)
        group_hours  = total_hours - theory_hours

        rag_content = dedent(f"""
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
        """).strip()

        return (
            [SystemMessage(content=GENERATION_SYSTEM_PROMPT)]
            + chat_history
            + [HumanMessage(content=rag_content)]
        )

    return RunnableLambda(retrieve_and_build_messages) | structured_llm


# ─── 대화형 정보 수집 ─────────────────────────────────────────────────────────

COLLECTION_SYSTEM_PROMPT = dedent("""
    당신은 기업 AI 교육 커리큘럼 설계를 위한 정보 수집 어시스턴트다.
    아래 항목을 자연스러운 대화로 순서대로 수집해라.
    한 번에 1개씩 질문하고, 모든 항목이 수집되면 마지막에 "[정보 수집 완료]"를 출력해라.

    수집 항목:
    - 회사명 또는 팀 이름
    - 교육 목표
    - 교육 대상자
    - 현재 AI 활용 수준 (입문/초급/중급)
    - 총 교육 기간 (일수)
    - 하루 교육 시간 (시간)
    - 원하는 핵심 주제
    - 반영해야 할 조건 또는 제한사항
    - AX Compass 진단 결과: 6개 유형별 인원수 (균형형, 이해형, 과신형, 실행형, 판단형, 조심형)
""").strip()


def run_collection_chat(llm: ChatOpenAI) -> list:
    messages = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)]
    response = llm.invoke(messages)
    messages.append(response)
    print(f"\n어시스턴트: {response.content}\n")

    while "[정보 수집 완료]" not in response.content:
        user_input = input("나: ").strip()
        if not user_input:
            continue
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(response)
        print(f"\n어시스턴트: {response.content}\n")

    return messages


def extract_collected_info(llm: ChatOpenAI, messages: list) -> CollectedInfo:
    extract_llm = llm.with_structured_output(CollectedInfo)
    return extract_llm.invoke(
        messages + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
    )


def calculate_groups(info: CollectedInfo) -> dict:
    return {
        "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],
                    "count": info.count_balanced + info.count_learner},
        "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],
                    "count": info.count_overconfident + info.count_doer},
        "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"],
                    "count": info.count_analyst + info.count_cautious},
    }


# ─── 출력 ─────────────────────────────────────────────────────────────────────

def print_curriculum(curriculum: dict):
    print("\n" + "=" * 60)
    print(f"과정명: {curriculum['program_title']}")
    print(f"대상 요약: {curriculum['target_summary']}")

    print("\n[공통 이론 수업]")
    for i, s in enumerate(curriculum["theory_sessions"], 1):
        print(f"\n  {i}. {s['title']}  ({s['duration_hours']}시간)")
        for g in s["goals"]:
            print(f"     목표: {g}")
        for a in s["activities"]:
            print(f"     활동: {a}")

    print("\n[그룹별 맞춤 실습]")
    for group in curriculum["group_sessions"]:
        print(f"\n  ┌ {group['group_name']} | {group['target_types']} | {group['participant_count']}명")
        print(f"  │ 실습 포커스: {group['focus_description']}")
        for i, s in enumerate(group["sessions"], 1):
            print(f"\n  {i}. {s['title']}  ({s['duration_hours']}시간)")
            for g in s["goals"]:
                print(f"     목표: {g}")
            for a in s["activities"]:
                print(f"     활동: {a}")

    print("\n예상 결과")
    for o in curriculum["expected_outcomes"]:
        print(f"- {o}")
    print("\n참고 사항")
    for n in curriculum["notes"]:
        print(f"- {n}")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run_chatbot():
    load_env_file()

    print(dedent("""
        =============================================
        기업 교육용 AI 커리큘럼 설계 챗봇 (LangChain + VectorDB)
        =============================================
        AX Compass 진단 결과를 반영해 그룹별 맞춤형 커리큘럼 초안을 생성합니다.
    """).strip())

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = setup_vector_store()

    # Phase 1: 대화로 정보 수집
    messages = run_collection_chat(llm)

    # Phase 2: 구조화 추출 및 그룹 계산
    print("\n[처리 중] 입력 정보를 분석하는 중...")
    info   = extract_collected_info(llm, messages)
    groups = calculate_groups(info)
    total_hours = info.days * info.hours_per_day

    # Phase 3: RAG 기반 커리큘럼 생성
    print("\n[생성 중] 커리큘럼을 생성하고 있습니다...\n")
    chain = build_chain(vectorstore)
    result: CurriculumPlan = chain.invoke({
        "conversation": messages,
        "groups":       groups,
        "total_hours":  total_hours,
        "topic":        info.topic,
        "level":        info.level,
    })

    print_curriculum(result.model_dump())


if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as error:
        print(f"\n[오류] {error}")