import os
from textwrap import dedent
from typing import Literal

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# --- Config ---

BASE_DIR        = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_PATH        = os.path.join(BASE_DIR, "Data", "AXCompass.pdf")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_types"

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}

SECTION_MARKERS = {
    "강점":   ["강점", "장점"],
    "보완방향": ["보완 방향", "보완방향", "보완"],
    "대표태그": ["대표 태그", "태그"],
}


def detect_section_type(text: str) -> str:
    for section, keywords in SECTION_MARKERS.items():
        if any(kw in text for kw in keywords):
            return section
    return "일반"

COLLECTION_SYSTEM_PROMPT = dedent("""
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
""").strip()

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
""").strip()


# --- Pydantic 스키마 ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str
    complete: bool
    collected_info: dict | None = None


class GenerateRequest(BaseModel):
    messages: list[Message]
    collected_info: dict


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
    company_name:        str = Field(description="회사명 또는 팀 이름")
    goal:                str = Field(description="교육 목표")
    audience:            str = Field(description="교육 대상자")
    level:               str = Field(description="현재 AI 활용 수준")
    days:                int = Field(description="총 교육 기간 (일수)")
    hours_per_day:       int = Field(description="하루 교육 시간 (시간)")
    topic:               str = Field(description="원하는 핵심 주제")
    constraints:         str = Field(description="반영해야 할 조건 또는 제한사항")
    count_balanced:      int = Field(description="균형형 인원수")
    count_learner:       int = Field(description="이해형 인원수")
    count_overconfident: int = Field(description="과신형 인원수")
    count_doer:          int = Field(description="실행형 인원수")
    count_analyst:       int = Field(description="판단형 인원수")
    count_cautious:      int = Field(description="조심형 인원수")


# --- Auth ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not BACKEND_API_KEY:
        raise HTTPException(status_code=500, detail="서버에 BACKEND_API_KEY가 설정되지 않았습니다.")
    if key != BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


# --- 메시지 변환 헬퍼 ---

def to_lc_messages(messages: list[Message]) -> list:
    result = []
    for m in messages:
        if m.role == "user":
            result.append(HumanMessage(content=m.content))
        else:
            result.append(AIMessage(content=m.content))
    return result


# --- RAG Pipeline ---

def load_and_split_documents() -> list:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")

    print("[RAG] PDF 로드 중...")
    pages = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        for type_name, info in TYPE_INFO.items():
            if type_name in chunk.page_content:
                chunk.metadata.update({
                    "type_name":    type_name,
                    "group":        info["group"],
                    "english":      info["english"],
                    "section_type": detect_section_type(chunk.page_content),
                })
                break

    print(f"[RAG] {len(pages)}페이지 → {len(chunks)}개 청크 분할 완료")
    return chunks


def init_vector_store() -> Chroma:
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
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")
    return vectorstore


def retrieve_section(vectorstore: Chroma, type_names: list[str], section_type: str, query: str) -> str:
    docs = vectorstore.similarity_search(
        query,
        k=len(type_names) * 2,
        filter={"$and": [
            {"type_name":    {"$in": type_names}},
            {"section_type": section_type},
        ]},
    )
    return "\n\n".join(d.page_content for d in docs) if docs else "(검색 결과 없음)"


def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> dict:
    names = ', '.join(type_names)
    return {
        "strengths":    retrieve_section(vectorstore, type_names, "강점",   f"{names} 유형의 강점과 AI 활용 장점"),
        "improvements": retrieve_section(vectorstore, type_names, "보완방향", f"{names} 유형의 보완 방향과 개선점"),
    }


def build_generation_chain(vectorstore: Chroma):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation = input_dict["conversation"]
        info: CollectedInfo = input_dict["info"]
        groups = input_dict["groups"]
        total_hours  = info.days * info.hours_per_day
        theory_hours = round(total_hours * 0.65)
        group_hours  = total_hours - theory_hours

        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]
        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        chat_history = [m for m in conversation if not isinstance(m, SystemMessage)]

        def group_ctx_block(name, types, ctx):
            return dedent(f"""
                === {name} ({' · '.join(types)}) ===
                [강점]
                {ctx['strengths']}

                [보완방향]
                {ctx['improvements']}
            """).strip()

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
            {group_ctx_block(ga['name'], ga['types'], ctx_a)}

            {group_ctx_block(gb['name'], gb['types'], ctx_b)}

            {group_ctx_block(gc['name'], gc['types'], ctx_c)}
        """).strip()

        return (
            [SystemMessage(content=GENERATION_SYSTEM_PROMPT)]
            + chat_history
            + [HumanMessage(content=rag_content)]
        )

    return RunnableLambda(retrieve_and_build_messages) | structured_llm


# --- App 초기화 ---

_llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
_vectorstore = init_vector_store()
_gen_chain   = build_generation_chain(_vectorstore)

app = FastAPI(title="AI 커리큘럼 백엔드")


@app.get("/health")
def health(_: str = Security(verify_api_key)):
    return {"status": "ok", "chunks": _vectorstore._collection.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: str = Security(verify_api_key)):
    """정보 수집 대화. complete=True 시 collected_info도 함께 반환한다."""
    try:
        lc_messages = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        response = _llm.invoke(lc_messages)
        reply    = response.content
        complete = "[정보 수집 완료]" in reply

        collected_info = None
        if complete:
            extract_llm = _llm.with_structured_output(CollectedInfo)
            info: CollectedInfo = extract_llm.invoke(
                lc_messages + [AIMessage(content=reply)]
                + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
            )
            collected_info = info.model_dump()

        return ChatResponse(reply=reply, complete=complete, collected_info=collected_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate(req: GenerateRequest, _: str = Security(verify_api_key)):
    """collected_info를 받아 RAG 검색 + 커리큘럼 생성 (LLM 1회)."""
    try:
        info   = CollectedInfo(**req.collected_info)
        groups = {
            "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],
                        "count": info.count_balanced + info.count_learner},
            "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],
                        "count": info.count_overconfident + info.count_doer},
            "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"],
                        "count": info.count_analyst + info.count_cautious},
        }
        conversation = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        result: CurriculumPlan = _gen_chain.invoke({
            "conversation": conversation,
            "info":         info,
            "groups":       groups,
        })
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
