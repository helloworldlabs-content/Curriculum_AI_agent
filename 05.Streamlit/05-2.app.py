import json
import os
import base64
from textwrap import dedent

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from openai import OpenAI
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

SESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "title":      {"type": "string"},
        "goals":      {"type": "array", "items": {"type": "string"}},
        "activities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "goals", "activities"],
    "additionalProperties": False,
}

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


def extract_type_chunks(pdf_content: str) -> list:
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


def init_collection():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )
    chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        print("[VectorDB] 임베딩 생성 중...")
        chunks = extract_type_chunks(load_pdf_content())
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({collection.count()}개 청크)")
    return collection


def retrieve_type_context(collection, type_names: list) -> str:
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    results = collection.query(
        query_texts=[query], n_results=len(type_names),
        where={"type_name": {"$in": list(type_names)}},
        include=["documents"],
    )
    return "\n\n".join(results["documents"][0])


def run_rag(requirements: dict, groups: dict, collection) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]
    ctxs = {
        "a": retrieve_type_context(collection, ga["types"]),
        "b": retrieve_type_context(collection, gb["types"]),
        "c": retrieve_type_context(collection, gc["types"]),
    }

    system_prompt = dedent("""
        당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
        AX Compass 진단 결과와 유형별 특성을 바탕으로 맞춤형 교육 커리큘럼을 설계하라.
        - theory_sessions: 공통 이론 수업 4~6개
        - group_sessions: 3개 그룹별 실습 각 2~3개
        JSON만 반환. 마크다운 코드블록 없음.
    """).strip()

    user_prompt = dedent(f"""
        [기업 요구사항]
        회사/팀: {requirements['company_name']} | 목표: {requirements['goal']}
        대상자: {requirements['audience']} | 수준: {requirements['level']}
        기간: {requirements['duration']} | 주제: {requirements['topic']}
        제한사항: {requirements['constraints']}

        [그룹 구성]
        - {ga['name']} ({' · '.join(ga['types'])}): {ga['count']}명
        - {gb['name']} ({' · '.join(gb['types'])}): {gb['count']}명
        - {gc['name']} ({' · '.join(gc['types'])}): {gc['count']}명

        [AX Compass 유형 특성]
        그룹 A: {ctxs['a']}
        그룹 B: {ctxs['b']}
        그룹 C: {ctxs['c']}
    """).strip()

    schema = {
        "type": "json_schema", "name": "curriculum_plan_rag", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "program_title":     {"type": "string"},
                "target_summary":    {"type": "string"},
                "theory_sessions":   {"type": "array", "items": SESSION_SCHEMA},
                "group_sessions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "group_name":        {"type": "string"},
                        "target_types":      {"type": "string"},
                        "participant_count": {"type": "integer"},
                        "focus_description": {"type": "string"},
                        "sessions":          {"type": "array", "items": SESSION_SCHEMA},
                    },
                    "required": ["group_name", "target_types", "participant_count",
                                 "focus_description", "sessions"],
                    "additionalProperties": False,
                }},
                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                "notes":             {"type": "array", "items": {"type": "string"}},
            },
            "required": ["program_title", "target_summary", "theory_sessions",
                         "group_sessions", "expected_outcomes", "notes"],
            "additionalProperties": False,
        },
    }

    response = client.responses.create(
        model="gpt-4.1-mini", text={"format": schema},
        input=[{"role": "developer", "content": system_prompt},
               {"role": "user", "content": user_prompt}],
    )
    return json.loads(response.output_text.strip())


# --- App 초기화 ---
# Gunicorn --preload 옵션 사용 시 마스터 프로세스에서 한 번만 실행되어
# 모든 워커가 초기화된 컬렉션을 공유한다.

_collection = init_collection()

app = FastAPI(title="AI 커리큘럼 백엔드")


class GenerateRequest(BaseModel):
    requirements: dict
    groups: dict


@app.get("/health")
def health(_: str = Security(verify_api_key)):
    return {"status": "ok", "chunks": _collection.count()}


@app.post("/generate")
def generate(req: GenerateRequest, _: str = Security(verify_api_key)):
    try:
        return run_rag(req.requirements, req.groups, _collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
