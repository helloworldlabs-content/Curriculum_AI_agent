# ─────────────────────────────────────────────────────────────────────────────
# 표준 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
import hashlib                              # 청크 고유 ID 생성 (SHA-1 해시)
import os                                   # 환경변수 읽기, 파일 경로 처리
from datetime import datetime, timedelta, timezone  # JWT 만료 시간 계산
from textwrap import dedent                 # 멀티라인 문자열 앞의 들여쓰기 제거
from typing import Any, Literal             # 타입 힌트

# ─────────────────────────────────────────────────────────────────────────────
# 서드파티 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
import bcrypt                               # 비밀번호 단방향 해싱 및 검증
import jwt                                  # JWT(JSON Web Token) 발급 · 검증

# FastAPI: Python REST API 서버 프레임워크
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# LangChain: RAG 파이프라인 구성 라이브러리
from langchain_chroma import Chroma                             # 벡터 데이터베이스 (로컬 저장)
from langchain_community.document_loaders import (
    PyPDFLoader,              # PDF 파일을 페이지 단위로 읽어오는 로더
    UnstructuredExcelLoader,  # Excel 파일을 시트 단위로 읽어오는 로더
)
from langchain_core.documents import Document                   # 텍스트 + 메타데이터 단위
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # LLM 메시지 타입
from langchain_core.runnables import RunnableLambda             # 함수를 LCEL 체인에 연결
from langchain_openai import ChatOpenAI, OpenAIEmbeddings       # OpenAI LLM / 임베딩 모델
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 문서를 청크로 분할

# Pydantic: 데이터 유효성 검사 및 직렬화 (FastAPI 요청/응답 모델에 사용)
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 설정값 (환경변수 또는 기본값)
# ─────────────────────────────────────────────────────────────────────────────

# 파일 경로
# APP_BASE_DIR: Docker 환경에서는 /app, 로컬에서는 프로젝트 루트를 자동 계산
BASE_DIR       = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR       = os.path.join(BASE_DIR, "Data")           # 학습 문서 폴더
PDF_PATH       = os.path.join(DATA_DIR, "AXCompass.pdf")  # AX Compass 기준 문서
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")       # 벡터 DB 저장 경로

# 벡터 DB 컬렉션 이름 (버전이 바뀌면 이름을 올려서 재인덱싱 강제)
AX_COLLECTION_NAME         = "ax_compass_profiles_v3"
CURRICULUM_COLLECTION_NAME = "curriculum_examples_v3"

# 청크 크기 설정
# AX Compass: 유형 프로필이 길어서 청크를 크게 설정
# 커리큘럼 예시: 항목이 짧아 조금 더 작게 설정
AX_CHUNK_SIZE         = 900
AX_CHUNK_OVERLAP      = 120   # 청크 간 겹치는 부분 (문맥 연결을 위해)
CURRICULUM_CHUNK_SIZE    = 700
CURRICULUM_CHUNK_OVERLAP = 80

# 인증 관련
BACKEND_API_KEY    = os.getenv("BACKEND_API_KEY", "")
JWT_SECRET         = os.getenv("JWT_SECRET", BACKEND_API_KEY)  # JWT 서명 키 (미설정 시 API Key 사용)
JWT_EXPIRY_HOURS   = int(os.getenv("JWT_EXPIRY_HOURS", "24"))  # 토큰 유효 시간 (기본 24시간)
AUTH_USERNAME      = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH", "").strip("'\"")  # .env의 따옴표 제거

# AX Compass 6가지 유형 정보
# group: 그룹 배정 기준 (A/B/C), english: 영문명
TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM 프롬프트
# ─────────────────────────────────────────────────────────────────────────────

# 정보 수집 단계에서 LLM이 사용하는 시스템 프롬프트
# "[정보 수집 완료]" 문자열이 포함되면 수집이 끝난 것으로 판단
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

# 커리큘럼 생성 단계에서 LLM이 따르는 시스템 프롬프트
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


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 스키마 (API 요청/응답 및 LLM 구조화 출력에 사용)
# ─────────────────────────────────────────────────────────────────────────────

# 채팅 메시지 단위
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# /chat 엔드포인트 요청 모델
class ChatRequest(BaseModel):
    messages: list[Message]  # 지금까지의 대화 기록


# /chat 엔드포인트 응답 모델
class ChatResponse(BaseModel):
    reply: str                        # LLM 응답 텍스트
    complete: bool                    # 정보 수집 완료 여부
    collected_info: dict | None = None  # 완료 시 추출된 구조화 정보


# /generate 엔드포인트 요청 모델
class GenerateRequest(BaseModel):
    messages: list[Message]   # 전체 대화 기록
    collected_info: dict      # /chat에서 추출된 구조화 정보


# 커리큘럼 내 개별 세션 (이론 또는 실습 1회차)
class Session(BaseModel):
    title: str
    duration_hours: float
    goals: list[str]
    activities: list[str]


# 그룹별 실습 세션 묶음
class GroupSession(BaseModel):
    group_name: str
    target_types: str           # 대상 AX Compass 유형
    participant_count: int
    focus_description: str      # 이 그룹의 실습 포커스
    sessions: list[Session]


# LLM이 최종 출력하는 커리큘럼 전체 구조
class CurriculumPlan(BaseModel):
    program_title: str
    target_summary: str
    theory_sessions: list[Session]
    group_sessions: list[GroupSession]
    expected_outcomes: list[str]
    notes: list[str]


# 정보 수집 대화에서 추출되는 구조화 데이터
# Field(description=...) 는 LLM이 어떤 값을 채워야 하는지 알려주는 힌트
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


# ─────────────────────────────────────────────────────────────────────────────
# 인증 (JWT 기반)
# 흐름: 로그인 → 토큰 발급 → 이후 모든 API 요청에 토큰 첨부
# ─────────────────────────────────────────────────────────────────────────────

# 로그인 요청 모델
class LoginRequest(BaseModel):
    username: str
    password: str

# 로그인 성공 응답 모델
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Authorization: Bearer <token> 헤더를 자동으로 파싱하는 FastAPI 보안 스킴
_bearer = HTTPBearer()


# 로그인 성공 시 JWT 토큰을 생성한다.
def create_token(username: str) -> str:
    payload = {
        "sub": username,                                                      # 토큰 주체 (사용자명)
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),  # 만료 시각
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# 요청 헤더의 JWT 토큰을 검증하고 사용자명을 반환한다.
# 유효하지 않거나 만료된 토큰은 401 에러를 반환한다.
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 메시지 변환 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

# Pydantic Message 리스트를 LangChain 메시지 객체 리스트로 변환한다.
_MSG_CLS = {"user": HumanMessage, "assistant": AIMessage}

def to_lc_messages(messages: list[Message]) -> list:
    return [_MSG_CLS[m.role](content=m.content) for m in messages]


# ─────────────────────────────────────────────────────────────────────────────
# RAG 파이프라인
#
# 전체 흐름:
#   1. 문서 로드 (PDF / Excel)
#   2. 전처리 (텍스트 정규화, 구조 보존)
#   3. 청킹 (문서 종류별 크기 분리)
#   4. 메타데이터 주석 (chunk_id, content_hash 등)
#   5. 벡터 DB 저장 (AX Compass / 커리큘럼 예시 컬렉션 분리)
#   6. 검색 → 프롬프트 구성 → LLM 생성
# ─────────────────────────────────────────────────────────────────────────────

# ── 전처리 헬퍼 함수들 ────────────────────────────────────────────────────────

# 연속 공백을 하나로 줄이고, 빈 줄이 연속될 경우 하나만 남긴다.
def _clean_text(text: str) -> str:
    normalized_lines = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())   # 탭·연속 공백 → 단일 공백
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif normalized_lines and not previous_blank:
            normalized_lines.append("")     # 빈 줄은 최대 1개
            previous_blank = True
    return "\n".join(normalized_lines).strip()


# 텍스트의 SHA-1 해시 앞 12자를 반환한다. 청크 중복 감지에 사용.
def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# PyPDFLoader의 page 메타데이터(0-indexed)를 1-indexed 페이지 번호로 변환한다.
def _page_number_from_metadata(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return None


# 텍스트의 첫 번째 비어있지 않은 줄을 섹션 제목으로 반환한다. 없으면 fallback 사용.
def _extract_section_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:120]  # 너무 긴 제목은 120자로 자름
    return fallback


# 텍스트에 AX Compass 유형명이 포함되면 해당 유형의 메타데이터를 추가한다.
def _tag_ax_type(metadata: dict[str, Any], text: str) -> None:
    for type_name, info in TYPE_INFO.items():
        if type_name in text:
            metadata.update({
                "type_name": type_name,
                "group":     info["group"],
                "english":   info["english"],
            })
            return  # 첫 번째로 매칭된 유형만 태깅


# 텍스트 키워드를 기반으로 커리큘럼 콘텐츠 유형을 추론한다.
# 이 값은 메타데이터로 저장되어 나중에 필터 검색에 활용 가능하다.
def _infer_curriculum_content_type(text: str) -> str:
    lowered = text.lower()
    if any(kw in lowered for kw in ("activity", "activities", "exercise", "workshop", "practice")):
        return "activity_plan"
    if any(kw in lowered for kw in ("goal", "goals", "objective", "learning outcome")):
        return "learning_goal"
    if any(kw in lowered for kw in ("case", "template", "agenda", "session")):
        return "session_plan"
    return "curriculum_reference"


# Excel 시트 내용을 LLM이 이해하기 쉬운 구조화 텍스트로 변환한다.
# 탭 구분 행 → [row] 태그, 키:값 형식 → [field] 태그, 나머지 → [entry] 태그.
def _format_excel_content(text: str) -> str:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in raw_line:
            # 탭으로 구분된 셀 → 행 형태로 표현
            cells = [cell.strip() for cell in raw_line.split("\t") if cell.strip()]
            if cells:
                rows.append("[row] " + " | ".join(cells))
                continue
        if ":" in line:
            # "키: 값" 형식 → 필드 형태로 표현
            key, value = line.split(":", 1)
            if key.strip() and value.strip():
                rows.append(f"[field] {key.strip()}: {value.strip()}")
                continue
        rows.append(f"[entry] {' '.join(line.split())}")
    return "\n".join(rows).strip()


# 청킹 전에 메타데이터를 텍스트 헤더로 본문 앞에 삽입한다.
#
# 예시:
#     [doc_type] ax_compass
#     [source_name] AXCompass.pdf
#     [type_name] 균형형
#     [content]
#     판단과 실행이 고르게 나타나며 ...
#
# 이렇게 하면 청크가 잘려도 문서 출처와 유형 정보가 함께 임베딩되어
# 검색 정확도가 올라간다.
def _build_structured_document(content: str, metadata: dict[str, Any]) -> Document:
    header_fields = [
        "doc_type", "source_name", "course_name", "sheet_name",
        "page_number", "section_title", "module_name", "content_type",
        "type_name", "group", "english",
    ]
    header_lines = []
    for field_name in header_fields:
        value = metadata.get(field_name)
        if value not in (None, ""):
            header_lines.append(f"[{field_name}] {value}")
    header_lines.append("[content]")
    header_lines.append(content)
    return Document(page_content="\n".join(header_lines).strip(), metadata=metadata)


# 각 청크에 고유 ID와 해시를 부여한다.
#
# - chunk_index: 같은 출처(파일 + 페이지 + 섹션) 내 청크 순서
# - content_hash: 내용 기반 12자리 해시 (중복 감지 및 증분 인덱싱에 활용)
# - chunk_id: "{파일명}:{순서}:{해시}" 형식의 전역 고유 ID
def _annotate_chunks(chunks: list[Document]) -> list[Document]:
    source_counters: dict[tuple[str, str, str], int] = {}
    for chunk in chunks:
        # 출처 키: (파일경로, 페이지번호, 섹션제목) 조합으로 청크 순서 카운트
        source_key = (
            str(chunk.metadata.get("source_file", "")),
            str(chunk.metadata.get("page_number", "")),
            str(chunk.metadata.get("section_title", "")),
        )
        chunk_index = source_counters.get(source_key, 0)
        source_counters[source_key] = chunk_index + 1

        content_hash = _short_hash(chunk.page_content)
        chunk.metadata.update({
            "chunk_index":  chunk_index,
            "content_hash": content_hash,
            "chunk_id":     f"{chunk.metadata.get('source_name', 'doc')}:{chunk_index}:{content_hash}",
        })
        # AX Compass 청크는 청킹 후에도 유형 태그를 유지
        if chunk.metadata.get("doc_type") == "ax_compass":
            _tag_ax_type(chunk.metadata, chunk.page_content)
    return chunks


# 문서를 지정된 청크 크기로 분할하고 메타데이터 주석을 붙인다.
#
# separators 우선순위:
#     1. "\n[content]" : _build_structured_document가 삽입한 헤더/본문 경계
#     2. "\n\n"        : 단락 경계
#     3. "\n"          : 줄 경계
#     4. " "           : 단어 경계 (마지막 수단)
def _split_documents_with_strategy(
    docs: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n[content]", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    return _annotate_chunks(chunks)


# ── 문서 로드 함수들 ──────────────────────────────────────────────────────────

# PDF 파일 1개를 페이지별로 읽어 Document 리스트로 반환하는 공통 헬퍼.
# metadata_builder(page, body, fpath) 콜백으로 문서 종류별 메타데이터를 구성한다.
def _load_pdf_pages(fpath: str, metadata_builder) -> list[Document]:
    documents = []
    for page in PyPDFLoader(fpath).load():
        body = _clean_text(page.page_content)
        if not body:
            continue
        metadata = metadata_builder(page, body, fpath)
        documents.append(_build_structured_document(body, metadata))
    return documents


# AXCompass.pdf를 로드하고 페이지별로 전처리된 Document를 반환한다.
def load_ax_compass_documents() -> list[Document]:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")

    print("[RAG] AXCompass PDF 로드 중...")

    def _build_meta(page, body, fpath):
        source_file   = page.metadata.get("source", fpath)
        page_number   = _page_number_from_metadata(page.metadata.get("page"))
        section_title = _extract_section_title(body, f"ax_page_{page_number or 'unknown'}")
        metadata = {
            "doc_type":      "ax_compass",
            "source_file":   source_file,
            "source_name":   os.path.basename(source_file),
            "page_number":   page_number,
            "section_title": section_title,
            "content_type":  "reference_profile",
        }
        _tag_ax_type(metadata, body)  # 유형명이 포함된 페이지에 유형 메타데이터 추가
        return metadata

    return _load_pdf_pages(PDF_PATH, _build_meta)


# Data 폴더의 커리큘럼 예시 PDF(AXCompass.pdf 제외)를 모두 로드한다.
def load_curriculum_pdf_documents() -> list[Document]:
    documents = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".pdf") or fname == "AXCompass.pdf":
            continue

        fpath       = os.path.join(DATA_DIR, fname)
        course_name = os.path.splitext(fname)[0]
        print(f"[RAG] 커리큘럼 PDF 로드: {fname}")

        def _build_meta(page, body, fpath, _course=course_name):
            source_file   = page.metadata.get("source", fpath)
            page_number   = _page_number_from_metadata(page.metadata.get("page"))
            section_title = _extract_section_title(body, _course)
            return {
                "doc_type":      "curriculum_example",
                "source_file":   source_file,
                "source_name":   os.path.basename(source_file),
                "course_name":   _course,
                "page_number":   page_number,
                "section_title": section_title,
                "module_name":   section_title,
                "content_type":  _infer_curriculum_content_type(body),
            }

        documents.extend(_load_pdf_pages(fpath, _build_meta))

    return documents


# Data 폴더의 커리큘럼 예시 Excel 파일을 모두 로드한다.
def load_curriculum_excel_documents() -> list[Document]:
    documents = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".xlsx"):
            continue

        fpath       = os.path.join(DATA_DIR, fname)
        course_name = os.path.splitext(fname)[0]
        print(f"[RAG] 커리큘럼 Excel 로드: {fname}")

        excel_docs = UnstructuredExcelLoader(fpath).load()
        for sheet_index, doc in enumerate(excel_docs, start=1):
            body = _format_excel_content(doc.page_content)
            if not body:
                continue

            source_file = doc.metadata.get("source", fpath)
            # UnstructuredExcelLoader가 시트 이름을 page_name 또는 sheet_name으로 제공
            sheet_name  = (
                doc.metadata.get("page_name")
                or doc.metadata.get("sheet_name")
                or f"sheet_{sheet_index}"
            )
            module_name = _extract_section_title(body, sheet_name)
            metadata = {
                "doc_type":     "curriculum_example",
                "source_file":  source_file,
                "source_name":  os.path.basename(source_file),
                "course_name":  course_name,
                "sheet_name":   sheet_name,
                "section_title": module_name,
                "module_name":  module_name,
                "content_type": _infer_curriculum_content_type(body),
            }
            documents.append(_build_structured_document(body, metadata))

    return documents


# AX Compass 문서를 로드하고 청킹까지 완료한 청크 리스트를 반환한다.
def load_ax_compass_chunks() -> list[Document]:
    docs   = load_ax_compass_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=AX_CHUNK_SIZE,
        chunk_overlap=AX_CHUNK_OVERLAP,
    )
    print(f"[RAG] AX Compass 청크 수: {len(chunks)}")
    return chunks


# 커리큘럼 예시(PDF + Excel)를 로드하고 청킹까지 완료한 청크 리스트를 반환한다.
def load_curriculum_chunks() -> list[Document]:
    docs   = load_curriculum_pdf_documents() + load_curriculum_excel_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=CURRICULUM_CHUNK_SIZE,
        chunk_overlap=CURRICULUM_CHUNK_OVERLAP,
    )
    print(f"[RAG] 커리큘럼 예시 청크 수: {len(chunks)}")
    return chunks


# ── 벡터 DB 구축 ──────────────────────────────────────────────────────────────

# 벡터 DB 컬렉션에 저장된 청크 수를 반환한다.
def _collection_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


# 지정된 컬렉션 이름으로 Chroma 벡터 DB 인스턴스를 생성한다.
def _create_vector_store(embeddings: OpenAIEmbeddings, collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )


# 컬렉션이 비어 있을 때만 문서를 로드하고 인덱싱한다.
# 이미 데이터가 있으면 재인덱싱 없이 기존 컬렉션을 사용한다.
def _ensure_index(vectorstore: Chroma, loader, label: str) -> None:
    existing_count = _collection_count(vectorstore)
    if existing_count > 0:
        print(f"[VectorDB] 기존 {label} 컬렉션 로드 완료 ({existing_count}개 청크)")
        return

    print(f"[VectorDB] {label} 컬렉션 인덱싱 중...")
    chunks = loader()
    if chunks:
        vectorstore.add_documents(chunks)
    print(f"[VectorDB] {label} 컬렉션 완료 ({len(chunks)}개 청크)")


# AX Compass와 커리큘럼 예시를 별도 컬렉션으로 인덱싱하고 반환한다.
#
# 컬렉션을 분리하는 이유:
# - AX Compass: 유형 정보 전용, 검색 시 type_name 필터 활용
# - 커리큘럼 예시: 세션 구성 참고용, 필터 없이 의미 검색
def setup_vector_stores() -> dict[str, Chroma]:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    ax_store         = _create_vector_store(embeddings, AX_COLLECTION_NAME)
    curriculum_store = _create_vector_store(embeddings, CURRICULUM_COLLECTION_NAME)

    _ensure_index(ax_store,         load_ax_compass_chunks,  "AX Compass")
    _ensure_index(curriculum_store, load_curriculum_chunks,  "커리큘럼 예시")

    return {
        "ax_compass":         ax_store,
        "curriculum_examples": curriculum_store,
    }


# ── 검색 함수 ─────────────────────────────────────────────────────────────────

# 벡터 DB에서 유사도 검색을 수행하는 공통 헬퍼.
# filter를 넘기면 해당 조건에 맞는 청크만 검색한다.
def _retrieve(vectorstore: Chroma, query: str, k: int, filter: dict | None = None) -> list[Document]:
    search_kwargs: dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs).invoke(query)


# 특정 AX Compass 유형들의 특성 정보를 벡터 DB에서 검색해 반환한다.
# k = max(4, len(type_names) * 3): 유형 수에 비례해 넉넉하게 검색
# filter: 해당 유형 청크만 검색 (다른 유형 내용이 섞이지 않도록)
def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    docs  = _retrieve(vectorstore, query, max(4, len(type_names) * 3), {"type_name": {"$in": type_names}})
    return "\n\n".join(d.page_content for d in docs)


# 주제와 수준에 맞는 커리큘럼 예시를 벡터 DB에서 검색해 반환한다.
def retrieve_curriculum_examples(vectorstore: Chroma, query: str, k: int = 3) -> str:
    docs = _retrieve(vectorstore, query, k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── LCEL 체인 구성 ────────────────────────────────────────────────────────────

# RAG + 커리큘럼 생성 체인을 구성한다.
#
# 체인 구조:
#     입력 dict
#         ↓
#     retrieve_and_build_messages (RunnableLambda)
#         - 벡터 DB에서 유형별 특성 및 커리큘럼 예시 검색
#         - 대화 기록 + 검색 결과를 하나의 프롬프트로 조합
#         ↓
#     structured_llm (ChatOpenAI → CurriculumPlan)
#         - LLM이 CurriculumPlan 구조로 직접 출력
def build_chain(vectorstores: dict[str, Chroma]):
    llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(CurriculumPlan)  # Pydantic 모델로 직접 파싱

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation = input_dict["conversation"]
        info: CollectedInfo = input_dict["info"]
        groups       = input_dict["groups"]

        # 총 시간을 이론 65% / 그룹 실습 35%로 미리 계산해서 LLM에 전달
        # (LLM이 직접 계산하면 오차가 생길 수 있으므로 수치를 확정해서 넘김)
        total_hours  = info.days * info.hours_per_day
        theory_hours = round(total_hours * 0.65)
        group_hours  = total_hours - theory_hours

        ax_vectorstore         = vectorstores["ax_compass"]
        curriculum_vectorstore = vectorstores["curriculum_examples"]

        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]

        # 그룹별 유형 특성 검색
        ctx_a = retrieve_group_context(ax_vectorstore, ga["types"])
        ctx_b = retrieve_group_context(ax_vectorstore, gb["types"])
        ctx_c = retrieve_group_context(ax_vectorstore, gc["types"])

        # 주제 + 수준 기반 커리큘럼 예시 검색
        curriculum_query    = f"{info.topic} {info.level} 기업 AI 교육 커리큘럼"
        curriculum_examples = retrieve_curriculum_examples(curriculum_vectorstore, curriculum_query)

        # SystemMessage는 이미 GENERATION_SYSTEM_PROMPT로 대체되므로 대화 기록에서 제외
        chat_history = [m for m in conversation if not isinstance(m, SystemMessage)]

        # 검색 결과를 포함한 최종 생성 요청 메시지
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


# ─────────────────────────────────────────────────────────────────────────────
# 앱 초기화 (서버 시작 시 1회 실행)
# 벡터 DB 구축과 체인 생성은 시간이 걸리므로 모듈 로드 시점에 미리 완료한다.
# ─────────────────────────────────────────────────────────────────────────────

_llm          = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
_vectorstores = setup_vector_stores()   # AX Compass + 커리큘럼 예시 컬렉션 로드/인덱싱
_chain        = build_chain(_vectorstores)  # RAG 체인 준비

app = FastAPI(title="AI 커리큘럼 백엔드")


# ─────────────────────────────────────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

# 아이디·비밀번호를 검증하고 JWT 토큰을 발급한다.
@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if not AUTH_PASSWORD_HASH:
        raise HTTPException(status_code=500, detail="서버에 AUTH_PASSWORD_HASH가 설정되지 않았습니다.")
    if req.username != AUTH_USERNAME or not bcrypt.checkpw(req.password.encode(), AUTH_PASSWORD_HASH.encode()):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return TokenResponse(access_token=create_token(req.username))


# 서버 상태와 벡터 DB 청크 수를 반환한다.
@app.get("/health")
def health(_: str = Depends(verify_token)):
    collections = {name: _collection_count(store) for name, store in _vectorstores.items()}
    return {"status": "ok", "chunks": sum(collections.values()), "collections": collections}


# 정보 수집 대화를 처리한다.
# - LLM이 "[정보 수집 완료]"를 출력하면 complete=True를 반환한다.
# - 완료 시 대화 내용에서 CollectedInfo를 구조화 추출해 함께 반환한다.
#   (추출을 여기서 하는 이유: /generate에서 LLM 호출을 1번으로 줄이기 위해)
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: str = Depends(verify_token)):
    try:
        lc_messages = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        response    = _llm.invoke(lc_messages)
        reply       = response.content
        complete    = "[정보 수집 완료]" in reply

        collected_info = None
        if complete:
            # 수집 완료 시 대화 전체를 넘겨 구조화 추출 (with_structured_output → Pydantic 직접 파싱)
            extract_llm    = _llm.with_structured_output(CollectedInfo)
            info: CollectedInfo = extract_llm.invoke(
                lc_messages + [AIMessage(content=reply)]
                + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
            )
            collected_info = info.model_dump()

        return ChatResponse(reply=reply, complete=complete, collected_info=collected_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AX Compass 유형 인원수를 A/B/C 그룹으로 묶어 반환한다.
# TYPE_INFO의 group 필드(A/B/C)와 대응되며, 체인 입력에 그대로 사용된다.
def _build_groups(info: CollectedInfo) -> dict:
    return {
        "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],  "count": info.count_balanced + info.count_learner},
        "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],  "count": info.count_overconfident + info.count_doer},
        "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"], "count": info.count_analyst + info.count_cautious},
    }


# 수집된 정보를 바탕으로 RAG 검색 + 커리큘럼 생성을 수행한다 (LLM 1회 호출).
# collected_info는 /chat에서 이미 추출된 값을 그대로 받으므로
# 이 엔드포인트에서는 추출 LLM 호출 없이 생성만 수행한다.
@app.post("/generate")
def generate(req: GenerateRequest, _: str = Depends(verify_token)):
    try:
        info         = CollectedInfo(**req.collected_info)
        conversation = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        result: CurriculumPlan = _chain.invoke({
            "conversation": conversation,
            "info":         info,
            "groups":       _build_groups(info),
        })
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
