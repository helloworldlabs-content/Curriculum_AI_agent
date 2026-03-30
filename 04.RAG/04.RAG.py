import json
import os
import base64
from textwrap import dedent

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# ─── 경로 설정 ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PDF_PATH = os.path.join(BASE_DIR, "Data", "AXCompass.pdf")
PDF_CACHE_PATH = os.path.join(BASE_DIR, "Data", "ax_compass_full.txt")

# 벡터 DB 저장 경로 — 최초 실행 시 임베딩을 생성해서 여기에 저장한다.
# 이후 실행부터는 이 파일을 그대로 불러오므로 임베딩 API 호출이 발생하지 않는다.
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_types"


# ─── 환경 설정 ─────────────────────────────────────────────────────────────────

def load_env_file(env_path=ENV_PATH):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ─── RAG 1단계 : PDF 텍스트 추출 ─────────────────────────────────────────────
#
# pypdf는 이 PDF의 한글 폰트 인코딩을 처리하지 못하므로
# OpenAI API를 통해 텍스트를 추출하고 캐시 파일에 저장한다.
# 이후 실행에서는 캐시 파일을 읽어서 API 비용을 아낀다.

def load_pdf_content():
    """
    AXCompass.pdf 에서 텍스트를 추출해 반환한다.
    캐시 파일이 있으면 바로 읽고, 없으면 OpenAI API로 추출 후 저장한다.
    """
    if os.path.exists(PDF_CACHE_PATH):
        with open(PDF_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    print("[RAG] PDF에서 유형 정보를 추출하는 중...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY를 찾을 수 없습니다: {ENV_PATH}")

    client = OpenAI(api_key=api_key)

    with open(PDF_PATH, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "AXCompass.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_data}",
                    },
                    {
                        "type": "input_text",
                        "text": (
                            "이 PDF에 있는 모든 내용을 한국어로 정확하게 추출해줘. "
                            "각 유형의 강점, 보완 방향, 대표 태그를 포함해서 전부 마크다운 형식으로 출력해줘. "
                            "각 유형 섹션은 '## 번호) 유형명' 형식으로 시작해줘."
                        ),
                    },
                ],
            }
        ],
    )

    content = response.output_text

    os.makedirs(os.path.dirname(PDF_CACHE_PATH), exist_ok=True)
    with open(PDF_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return content


# ─── RAG 2단계 : 청크 분할 ───────────────────────────────────────────────────
#
# 벡터 DB에 저장하기 전에 PDF 전체 텍스트를 유형별 청크로 나눈다.
# 각 청크는 하나의 유형(균형형, 실행형 등)에 대한 설명 전체를 담는다.

# 유형별 섹션 시작 마커 (load_pdf_content가 마크다운으로 추출한 형식에 맞춤)
TYPE_MARKERS = {
    "균형형": "## 1) 균형형",
    "실행형": "## 2) 실행형",
    "판단형": "## 3) 판단형",
    "이해형": "## 4) 이해형",
    "과신형": "## 5) 과신형",
    "조심형": "## 6) 조심형",
}

# 유형별 그룹 및 영문명 메타데이터
TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


def extract_type_chunks(pdf_content):
    """
    PDF 전체 텍스트를 유형별로 분할해 청크 리스트를 반환한다.
    각 청크는 벡터 DB의 문서 하나가 된다.

    반환 형식:
    [
        {
            "id": "균형형",
            "text": "## 1) 균형형 (BALANCED) ...",
            "metadata": {"type_name": "균형형", "group": "A", "english": "BALANCED"}
        },
        ...
    ]
    """
    chunks = []

    for type_name, marker in TYPE_MARKERS.items():
        start = pdf_content.find(marker)
        if start == -1:
            continue

        # 다음 유형 섹션이 시작되는 위치를 현재 섹션의 끝으로 삼는다.
        end = len(pdf_content)
        for other_marker in TYPE_MARKERS.values():
            if other_marker == marker:
                continue
            pos = pdf_content.find(other_marker, start + len(marker))
            if pos != -1 and pos < end:
                end = pos

        text = pdf_content[start:end].strip()
        info = TYPE_INFO[type_name]

        chunks.append({
            "id": type_name,
            "text": text,
            "metadata": {
                "type_name": type_name,
                "group": info["group"],
                "english": info["english"],
            },
        })

    return chunks


# ─── RAG 3단계 : 벡터 DB 구축 ────────────────────────────────────────────────
#
# ChromaDB를 사용해 유형별 청크를 임베딩하고 로컬 파일로 저장한다.
# - 저장 위치: Curriculum_AI_agent/vectorDB/
# - 임베딩 모델: OpenAI text-embedding-3-small
# - 유사도 측정: cosine similarity
#
# 최초 실행 시에만 임베딩 API를 호출해 DB를 생성한다.
# 이후 실행에서는 VECTOR_DB_PATH에 저장된 파일을 불러와 바로 사용한다.

def setup_vector_db(pdf_content):
    """
    ChromaDB 컬렉션을 초기화하고 반환한다.

    - 데이터가 이미 있으면 기존 컬렉션을 그대로 로드
    - 비어있으면 PDF 청크를 임베딩해서 새로 저장
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY를 찾을 수 없습니다: {ENV_PATH}")

    # 영구 저장 클라이언트 — VECTOR_DB_PATH 폴더에 sqlite 파일로 저장된다.
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # OpenAI 임베딩 함수 설정
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},  # 코사인 유사도로 검색
    )

    # 이미 청크가 저장되어 있으면 재생성하지 않는다.
    if collection.count() > 0:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({collection.count()}개 청크)")
        return collection

    # 청크 분할 → 임베딩 → 저장
    print("[VectorDB] 임베딩 생성 중...")
    chunks = extract_type_chunks(pdf_content)

    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )

    print(f"[VectorDB] {len(chunks)}개 청크 저장 완료 → {VECTOR_DB_PATH}")
    return collection


# ─── RAG 4단계 : 시맨틱 검색 ─────────────────────────────────────────────────
#
# 단순 문자열 탐색(find)이 아닌 벡터 유사도 검색으로 컨텍스트를 가져온다.
# 검색 흐름:
#   1. 쿼리 텍스트(유형 이름 + 교육적 특성)를 임베딩 벡터로 변환
#   2. ChromaDB에서 코사인 유사도가 높은 청크를 검색
#   3. 메타데이터 필터($in)로 해당 그룹 유형만 좁혀서 반환

def retrieve_type_context(collection, type_names):
    """
    벡터 DB에서 특정 유형들의 설명을 시맨틱 검색으로 가져온다.

    Args:
        collection: ChromaDB 컬렉션 객체
        type_names: 검색할 유형 이름 리스트 (예: ["균형형", "이해형"])

    Returns:
        관련 청크들을 합친 문자열
    """
    # 자연어 쿼리로 임베딩 벡터를 생성해서 관련 청크를 찾는다.
    query = (
        f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    )

    results = collection.query(
        query_texts=[query],
        n_results=len(type_names),
        # 메타데이터 필터: 요청한 유형의 청크만 검색 대상으로 좁힌다.
        where={"type_name": {"$in": list(type_names)}},
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    return "\n\n".join(documents)


# ─── 입력 수집 ────────────────────────────────────────────────────────────────

def ask_question(prompt_text):
    return input(prompt_text).strip()


def collect_company_requirements():
    print("\n[기업 요구사항 입력]")
    print("질문에 답하면 AX Compass 진단 결과를 반영한 맞춤형 커리큘럼을 생성합니다.\n")

    return {
        "company_name": ask_question("1. 회사명 또는 팀 이름: "),
        "goal": ask_question("2. 교육 목표: "),
        "audience": ask_question("3. 교육 대상자: "),
        "level": ask_question("4. 현재 AI 활용 수준 (예: 입문, 초급, 중급): "),
        "duration": ask_question("5. 교육 기간 또는 총 시간: "),
        "topic": ask_question("6. 원하는 핵심 주제: "),
        "constraints": ask_question("7. 꼭 반영해야 할 조건 또는 제한사항: "),
    }


def collect_type_counts():
    print("\n[AX Compass 진단 결과 입력]")
    print("진단 검사 결과 유형별 인원수를 입력해주세요.\n")

    types = ["균형형", "이해형", "과신형", "실행형", "판단형", "조심형"]
    counts = {}

    for i, type_name in enumerate(types, 1):
        while True:
            value = ask_question(f"{i}. {type_name} 인원수: ")
            try:
                counts[type_name] = int(value)
                break
            except ValueError:
                print("   숫자를 입력해주세요.")

    return counts


def calculate_groups(type_counts):
    return {
        "group_a": {
            "name": "그룹 A",
            "types": ["균형형", "이해형"],
            "count": type_counts["균형형"] + type_counts["이해형"],
        },
        "group_b": {
            "name": "그룹 B",
            "types": ["과신형", "실행형"],
            "count": type_counts["과신형"] + type_counts["실행형"],
        },
        "group_c": {
            "name": "그룹 C",
            "types": ["판단형", "조심형"],
            "count": type_counts["판단형"] + type_counts["조심형"],
        },
    }


# ─── 요약 출력 ────────────────────────────────────────────────────────────────

def print_requirements_summary(requirements, type_counts, groups):
    print("\n[입력한 요구사항 요약]")
    print(f"- 회사/팀: {requirements['company_name']}")
    print(f"- 교육 목표: {requirements['goal']}")
    print(f"- 교육 대상자: {requirements['audience']}")
    print(f"- 현재 수준: {requirements['level']}")
    print(f"- 교육 기간: {requirements['duration']}")
    print(f"- 핵심 주제: {requirements['topic']}")
    print(f"- 제한사항: {requirements['constraints']}")

    total = sum(type_counts.values())
    print(f"\n[진단 결과 요약] 총 {total}명")
    for type_name, count in type_counts.items():
        print(f"  - {type_name}: {count}명")

    print("\n[그룹 구성]")
    for group in groups.values():
        types_str = " · ".join(group["types"])
        print(f"  - {group['name']} ({types_str}): {group['count']}명")


# ─── 프롬프트 빌더 ────────────────────────────────────────────────────────────

def build_system_prompt():
    return dedent("""
        당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
        AX Compass 진단 결과와 유형별 특성을 바탕으로 맞춤형 교육 커리큘럼을 설계하라.

        커리큘럼 구조:
        - theory_sessions: 모든 참가자가 동일하게 수강하는 공통 이론 수업. 4개 이상 6개 이하.
        - group_sessions: 3개 그룹이 각각 다른 실습 또는 프로젝트를 진행. 각 그룹당 2개 이상 3개 이하.

        반드시 아래 규칙을 지켜라.
        1. 출력은 반드시 JSON 하나만 반환한다.
        2. 설명 문장, 코드블록 마크다운, 인사말을 추가하지 않는다.
        3. 그룹별 실습은 해당 유형의 강점을 활용하고 보완 방향을 실습 활동에 녹인다.
        4. 각 회차는 title, goals, activities를 포함한다.
        5. 기업 교육답게 실무 적용 중심으로 구성한다.
        6. notes에는 유형별 특성과 기업 제한사항을 반영한 주의사항을 작성한다.
    """).strip()


def build_user_prompt(requirements, groups, type_contexts):
    """
    RAG의 Augment 단계:
    벡터 DB에서 검색한 유형 설명(type_contexts)을 프롬프트에 포함한다.
    """
    group_a = groups["group_a"]
    group_b = groups["group_b"]
    group_c = groups["group_c"]

    return dedent(f"""
        다음 기업 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

        [기업 요구사항]
        회사/팀: {requirements['company_name']}
        교육 목표: {requirements['goal']}
        교육 대상자: {requirements['audience']}
        현재 AI 활용 수준: {requirements['level']}
        교육 기간 또는 총 시간: {requirements['duration']}
        원하는 핵심 주제: {requirements['topic']}
        꼭 반영해야 할 조건 또는 제한사항: {requirements['constraints']}

        [그룹 구성]
        - {group_a['name']} ({' · '.join(group_a['types'])}): {group_a['count']}명
        - {group_b['name']} ({' · '.join(group_b['types'])}): {group_b['count']}명
        - {group_c['name']} ({' · '.join(group_c['types'])}): {group_c['count']}명

        [AX Compass 유형별 특성 — 벡터 DB 시맨틱 검색 결과]
        아래는 AXCompass.pdf에서 시맨틱 검색으로 가져온 각 유형의 공식 설명이다.
        이 내용을 반드시 반영해서 그룹별 실습을 설계하라.

        === 그룹 A 유형 설명 ({' · '.join(group_a['types'])}) ===
        {type_contexts['group_a']}

        === 그룹 B 유형 설명 ({' · '.join(group_b['types'])}) ===
        {type_contexts['group_b']}

        === 그룹 C 유형 설명 ({' · '.join(group_c['types'])}) ===
        {type_contexts['group_c']}

        [설계 요구사항]
        - 이론 수업은 3개 그룹 모두 동일하게 수강
        - 그룹 실습은 각 유형의 강점을 살리고 보완 방향을 실습 활동에 녹여서 구성
        - 초보자도 따라올 수 있도록 난이도를 조절
        - 실무에 바로 적용 가능한 활동 중심으로 구성
    """).strip()


# ─── JSON 스키마 ──────────────────────────────────────────────────────────────

def _session_item_schema():
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "goals": {"type": "array", "items": {"type": "string"}},
            "activities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "goals", "activities"],
        "additionalProperties": False,
    }


def build_curriculum_schema():
    return {
        "type": "json_schema",
        "name": "curriculum_plan_rag",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "program_title": {"type": "string"},
                "target_summary": {"type": "string"},
                "theory_sessions": {
                    "type": "array",
                    "items": _session_item_schema(),
                },
                "group_sessions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_name": {"type": "string"},
                            "target_types": {"type": "string"},
                            "participant_count": {"type": "integer"},
                            "focus_description": {"type": "string"},
                            "sessions": {
                                "type": "array",
                                "items": _session_item_schema(),
                            },
                        },
                        "required": [
                            "group_name",
                            "target_types",
                            "participant_count",
                            "focus_description",
                            "sessions",
                        ],
                        "additionalProperties": False,
                    },
                },
                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "program_title",
                "target_summary",
                "theory_sessions",
                "group_sessions",
                "expected_outcomes",
                "notes",
            ],
            "additionalProperties": False,
        },
    }


# ─── API 호출 ─────────────────────────────────────────────────────────────────

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


def generate_curriculum(requirements, groups, collection):
    """
    RAG 파이프라인 전체를 실행한다.

    ① Retrieve : 벡터 DB에서 그룹별 유형 설명을 시맨틱 검색
    ② Augment  : 검색된 내용을 프롬프트에 포함
    ③ Generate : OpenAI API로 커리큘럼 JSON 생성
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY를 찾을 수 없습니다: {ENV_PATH}")

    client = OpenAI(api_key=api_key)

    # ① Retrieve: 벡터 DB에서 그룹별 유형 설명 시맨틱 검색
    print("[RAG] 벡터 DB에서 유형별 컨텍스트를 검색하는 중...")
    type_contexts = {
        "group_a": retrieve_type_context(collection, groups["group_a"]["types"]),
        "group_b": retrieve_type_context(collection, groups["group_b"]["types"]),
        "group_c": retrieve_type_context(collection, groups["group_c"]["types"]),
    }

    # ② Augment + ③ Generate
    response = client.responses.create(
        model="gpt-4.1-mini",
        text={"format": build_curriculum_schema()},
        input=[
            {"role": "developer", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(requirements, groups, type_contexts)},
        ],
    )

    raw_text = extract_text_from_response(response)
    if not raw_text:
        raise ValueError("모델 응답에서 텍스트를 읽지 못했습니다.")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "모델 응답이 JSON 형식이 아니어서 해석할 수 없습니다.\n"
            f"응답 내용: {raw_text}"
        ) from error


# ─── 결과 출력 ────────────────────────────────────────────────────────────────

def print_curriculum(curriculum):
    print("\n" + "=" * 60)
    print("[추천 커리큘럼 초안]")
    print("=" * 60)
    print(f"\n과정명: {curriculum['program_title']}")
    print(f"대상 요약: {curriculum['target_summary']}")

    print("\n" + "─" * 60)
    print("[공통 이론 수업] — 모든 그룹 동일하게 진행")
    print("─" * 60)

    for index, session in enumerate(curriculum["theory_sessions"], start=1):
        print(f"\n  {index}. {session['title']}")
        print("     목표")
        for goal in session["goals"]:
            print(f"     - {goal}")
        print("     활동")
        for activity in session["activities"]:
            print(f"     - {activity}")

    print("\n" + "─" * 60)
    print("[그룹별 맞춤 실습] — 각 그룹이 별도로 진행")
    print("─" * 60)

    for group in curriculum["group_sessions"]:
        print(f"\n  ┌ {group['group_name']} | {group['target_types']} | {group['participant_count']}명")
        print(f"  │ 실습 포커스: {group['focus_description']}")

        for index, session in enumerate(group["sessions"], start=1):
            print(f"\n  {index}. {session['title']}")
            print("     목표")
            for goal in session["goals"]:
                print(f"     - {goal}")
            print("     활동")
            for activity in session["activities"]:
                print(f"     - {activity}")

    print("\n" + "─" * 60)
    print("예상 결과")
    for outcome in curriculum["expected_outcomes"]:
        print(f"- {outcome}")

    print("\n참고 사항")
    for note in curriculum["notes"]:
        print(f"- {note}")


# ─── 메인 흐름 ────────────────────────────────────────────────────────────────

def run_chatbot():
    load_env_file()

    print(
        dedent("""
            =========================================
            기업 교육용 AI 커리큘럼 설계 챗봇 (RAG + VectorDB 버전)
            =========================================
            AX Compass 진단 결과를 반영해 그룹별 맞춤형 커리큘럼 초안을 생성합니다.
        """).strip()
    )

    # 1) PDF 텍스트 추출 (캐시 우선)
    pdf_content = load_pdf_content()

    # 2) 벡터 DB 초기화
    #    최초 실행: 임베딩 생성 → vectorDB/ 폴더에 저장
    #    이후 실행: vectorDB/ 폴더에서 로드
    collection = setup_vector_db(pdf_content)

    # 3) 사용자 입력 수집
    requirements = collect_company_requirements()
    type_counts = collect_type_counts()
    groups = calculate_groups(type_counts)

    # 4) 요약 출력
    print_requirements_summary(requirements, type_counts, groups)

    # 5) 커리큘럼 생성
    print("\n[생성 중]")
    print("진단 결과와 요구사항을 바탕으로 맞춤형 커리큘럼을 생성하고 있습니다...\n")

    curriculum = generate_curriculum(requirements, groups, collection)

    # 6) 결과 출력
    print_curriculum(curriculum)


if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as error:
        print("\n[오류]")
        print(error)
