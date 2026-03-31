import os
import base64
from textwrap import dedent

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

# --- Config ---

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH        = os.path.join(BASE_DIR, "Data", "AXCompass.pdf")
PDF_CACHE_PATH  = os.path.join(BASE_DIR, "Data", "ax_compass_full.txt")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_types"

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

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

SYSTEM_PROMPT = dedent("""
    당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
    AX Compass 진단 결과와 유형별 특성을 바탕으로 맞춤형 교육 커리큘럼을 설계하라.

    커리큘럼 구조:
    - theory_sessions: 모든 참가자가 동일하게 수강하는 공통 이론 수업. 4개 이상 6개 이하.
    - group_sessions: 3개 그룹이 각각 다른 실습을 진행. 각 그룹당 2개 이상 3개 이하.

    규칙:
    1. 그룹별 실습은 해당 유형의 강점을 활용하고 보완 방향을 실습 활동에 녹인다.
    2. 각 회차는 title, goals, activities를 포함한다.
    3. 기업 교육답게 실무 적용 중심으로 구성한다.
    4. notes에는 유형별 특성과 기업 제한사항을 반영한 주의사항을 작성한다.
""").strip()


# --- Pydantic 스키마 (Structured Output + API 요청) ---

class Session(BaseModel):
    title: str
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


class GenerateRequest(BaseModel):
    requirements: dict
    groups: dict


# --- Auth ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not BACKEND_API_KEY:
        raise HTTPException(status_code=500, detail="서버에 BACKEND_API_KEY가 설정되지 않았습니다.")
    if key != BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


# --- RAG Pipeline ---

def load_pdf_content() -> str:
    if os.path.exists(PDF_CACHE_PATH):
        with open(PDF_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(PDF_PATH, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": [
            {"type": "input_file", "filename": "AXCompass.pdf",
             "file_data": f"data:application/pdf;base64,{pdf_data}"},
            {"type": "input_text", "text":
             "이 PDF의 모든 내용을 한국어 마크다운으로 추출해줘. "
             "각 유형의 강점, 보완 방향, 대표 태그를 포함하고 "
             "각 유형 섹션은 '## 번호) 유형명' 형식으로 시작해줘."},
        ]}],
    )
    content = response.output_text
    os.makedirs(os.path.dirname(PDF_CACHE_PATH), exist_ok=True)
    with open(PDF_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def extract_type_chunks(pdf_content: str) -> list[dict]:
    chunks = []
    for type_name, marker in TYPE_MARKERS.items():
        start = pdf_content.find(marker)
        if start == -1:
            continue
        end = min(
            (pdf_content.find(om, start + len(marker))
             for om in TYPE_MARKERS.values() if om != marker
             if pdf_content.find(om, start + len(marker)) != -1),
            default=len(pdf_content),
        )
        info = TYPE_INFO[type_name]
        chunks.append({
            "id": type_name,
            "text": pdf_content[start:end].strip(),
            "metadata": {"type_name": type_name, "group": info["group"], "english": info["english"]},
        })
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
        chunks = extract_type_chunks(load_pdf_content())
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
        vectorstore.add_documents(docs, ids=[c["id"] for c in chunks])
        print(f"[VectorDB] {len(docs)}개 청크 저장 완료")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")
    return vectorstore


def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    docs = vectorstore.similarity_search(
        query, k=len(type_names),
        filter={"type_name": {"$in": type_names}},
    )
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vectorstore: Chroma):
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        req    = input_dict["requirements"]
        groups = input_dict["groups"]
        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]

        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        user_content = dedent(f"""
            다음 기업 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

            [기업 요구사항]
            회사/팀: {req['company_name']} | 목표: {req['goal']}
            대상자: {req['audience']} | 수준: {req['level']}
            기간: {req['duration']} | 주제: {req['topic']}
            제한사항: {req['constraints']}

            [그룹 구성]
            - {ga['name']} ({' · '.join(ga['types'])}): {ga['count']}명
            - {gb['name']} ({' · '.join(gb['types'])}): {gb['count']}명
            - {gc['name']} ({' · '.join(gc['types'])}): {gc['count']}명

            [AX Compass 유형별 특성 — 벡터 DB 시맨틱 검색 결과]
            === 그룹 A ({' · '.join(ga['types'])}) ===
            {ctx_a}

            === 그룹 B ({' · '.join(gb['types'])}) ===
            {ctx_b}

            === 그룹 C ({' · '.join(gc['types'])}) ===
            {ctx_c}
        """).strip()

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    return RunnableLambda(retrieve_and_build_messages) | structured_llm


# --- App 초기화 ---
# Gunicorn --preload 옵션 사용 시 마스터 프로세스에서 한 번만 실행된다.

_vectorstore = init_vector_store()
_chain = build_chain(_vectorstore)

app = FastAPI(title="AI 커리큘럼 백엔드")


@app.get("/health")
def health(_: str = Security(verify_api_key)):
    return {"status": "ok", "chunks": _vectorstore._collection.count()}


@app.post("/generate")
def generate(req: GenerateRequest, _: str = Security(verify_api_key)):
    try:
        result: CurriculumPlan = _chain.invoke({
            "requirements": req.requirements,
            "groups": req.groups,
        })
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
