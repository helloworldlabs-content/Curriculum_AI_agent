import os
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas import (
    Message, CollectedInfo, CurriculumPlan,
)

# --- 경로 설정 ---

BASE_DIR       = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR       = os.path.join(BASE_DIR, "Data")
PDF_PATH       = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_v2"

# AX Compass 6가지 유형 정보
TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}

# --- 프롬프트 ---

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
    6. [커리큘럼 예시 참고 자료]가 제공되는 경우, 아래 기준으로 참고하되 내용을 그대로 복사하지 마라:
       - 세션 제목과 구성 방식 (이론→실습 흐름, 주제 전개 순서 등)
       - 활동 유형 (워크숍, 실습, 토론, 케이스 스터디 등)
       - 강의 내용의 깊이와 수준 (용어, 난이도, 현업 적용 방식)
""").strip()


# --- 메시지 변환 헬퍼 ---

_MSG_CLS = {"user": HumanMessage, "assistant": AIMessage}

def to_lc_messages(messages: list[Message]) -> list:
    """Pydantic Message 리스트를 LangChain 메시지 객체 리스트로 변환한다."""
    return [_MSG_CLS[m.role](content=m.content) for m in messages]


# --- 문서 로드 및 청킹 ---

def load_and_split_documents() -> list[Document]:
    all_docs = []

    # AX Compass PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")
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

    # AX Compass 청크에 유형 메타데이터 태깅
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


# --- 벡터 DB ---

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
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")
    return vectorstore


# --- 검색 함수 ---

def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    """특정 AX Compass 유형들의 특성 정보를 벡터 DB에서 검색해 반환한다."""
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
    """주제와 수준에 맞는 커리큘럼 예시를 벡터 DB에서 검색해 반환한다."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": {"$eq": "curriculum_example"}},
        },
    )
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# --- LCEL 체인 ---

def build_chain(vectorstore: Chroma):
    """RAG 검색 + 커리큘럼 생성 LCEL 체인을 구성한다."""
    llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation = input_dict["conversation"]
        info: CollectedInfo = input_dict["info"]
        groups = input_dict["groups"]

        # 총 시간을 이론 65% / 그룹 실습 35%로 미리 계산해서 LLM에 전달
        total_hours  = info.days * info.hours_per_day
        theory_hours = round(total_hours * 0.65)
        group_hours  = total_hours - theory_hours

        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]
        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        curriculum_query    = f"{info.topic} {info.level} 기업 AI 교육 커리큘럼"
        curriculum_examples = retrieve_curriculum_examples(vectorstore, curriculum_query)

        # SystemMessage는 이미 GENERATION_SYSTEM_PROMPT로 대체되므로 대화 기록에서 제외
        chat_history = [m for m in conversation if not isinstance(m, SystemMessage)]

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
