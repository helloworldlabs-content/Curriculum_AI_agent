import os
import base64
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel


# ─── 경로 설정 ────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH       = os.path.join(BASE_DIR, ".env")
PDF_PATH       = os.path.join(BASE_DIR, "Data", "AXCompass.pdf")
PDF_CACHE_PATH = os.path.join(BASE_DIR, "Data", "ax_compass_full.txt")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_types"

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


# ─── Pydantic 스키마 (Structured Output) ─────────────────────────────────────

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


# ─── RAG 1단계: PDF 텍스트 추출 ──────────────────────────────────────────────

def load_pdf_content() -> str:
    if os.path.exists(PDF_CACHE_PATH):
        with open(PDF_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    print("[RAG] PDF에서 유형 정보를 추출하는 중...")
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
             "이 PDF에 있는 모든 내용을 한국어로 정확하게 추출해줘. "
             "각 유형의 강점, 보완 방향, 대표 태그를 포함해서 전부 마크다운 형식으로 출력해줘. "
             "각 유형 섹션은 '## 번호) 유형명' 형식으로 시작해줘."},
        ]}],
    )
    content = response.output_text
    os.makedirs(os.path.dirname(PDF_CACHE_PATH), exist_ok=True)
    with open(PDF_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return content


# ─── RAG 2단계: 청크 분할 ────────────────────────────────────────────────────

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


# ─── RAG 3단계: 벡터 DB 구축 (LangChain Chroma) ──────────────────────────────

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
        chunks = extract_type_chunks(load_pdf_content())
        docs = [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]
        vectorstore.add_documents(docs, ids=[c["id"] for c in chunks])
        print(f"[VectorDB] {len(docs)}개 청크 저장 완료 → {VECTOR_DB_PATH}")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")

    return vectorstore


# ─── RAG 4단계: 시맨틱 검색 ──────────────────────────────────────────────────

def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    docs = vectorstore.similarity_search(
        query, k=len(type_names),
        filter={"type_name": {"$in": type_names}},
    )
    return "\n\n".join(d.page_content for d in docs)


# ─── RAG 5단계: LCEL 체인 구성 ───────────────────────────────────────────────

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


def build_chain(vectorstore: Chroma):
    """
    LCEL 체인:
    ① RunnableLambda: 벡터 DB 검색(Retrieve) + 메시지 구성(Augment)
    ② ChatOpenAI.with_structured_output: 커리큘럼 생성(Generate)
    """
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

        # ① Retrieve
        print("[RAG] 벡터 DB에서 유형별 컨텍스트를 검색하는 중...")
        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        # ② Augment
        user_content = dedent(f"""
            다음 기업 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

            [기업 요구사항]
            회사/팀: {req['company_name']}
            교육 목표: {req['goal']}
            교육 대상자: {req['audience']}
            현재 AI 활용 수준: {req['level']}
            교육 기간 또는 총 시간: {req['duration']}
            원하는 핵심 주제: {req['topic']}
            꼭 반영해야 할 조건 또는 제한사항: {req['constraints']}

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


# ─── 입력 수집 ────────────────────────────────────────────────────────────────

def ask_question(prompt_text: str) -> str:
    return input(prompt_text).strip()


def collect_company_requirements() -> dict:
    print("\n[기업 요구사항 입력]")
    print("질문에 답하면 AX Compass 진단 결과를 반영한 맞춤형 커리큘럼을 생성합니다.\n")
    return {
        "company_name": ask_question("1. 회사명 또는 팀 이름: "),
        "goal":         ask_question("2. 교육 목표: "),
        "audience":     ask_question("3. 교육 대상자: "),
        "level":        ask_question("4. 현재 AI 활용 수준 (예: 입문, 초급, 중급): "),
        "duration":     ask_question("5. 교육 기간 또는 총 시간: "),
        "topic":        ask_question("6. 원하는 핵심 주제: "),
        "constraints":  ask_question("7. 꼭 반영해야 할 조건 또는 제한사항: "),
    }


def collect_type_counts() -> dict:
    print("\n[AX Compass 진단 결과 입력]")
    print("진단 검사 결과 유형별 인원수를 입력해주세요.\n")
    types = ["균형형", "이해형", "과신형", "실행형", "판단형", "조심형"]
    counts = {}
    for i, type_name in enumerate(types, 1):
        while True:
            value = ask_question(f"{i}. {type_name} 인원수: ")
            if value.isdigit():
                counts[type_name] = int(value)
                break
            print("   숫자를 입력해주세요.")
    return counts


def calculate_groups(type_counts: dict) -> dict:
    return {
        "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],
                    "count": type_counts["균형형"] + type_counts["이해형"]},
        "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],
                    "count": type_counts["과신형"] + type_counts["실행형"]},
        "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"],
                    "count": type_counts["판단형"] + type_counts["조심형"]},
    }


# ─── 출력 ─────────────────────────────────────────────────────────────────────

def print_requirements_summary(requirements, type_counts, groups):
    print("\n[입력한 요구사항 요약]")
    for label, key in [("회사/팀", "company_name"), ("교육 목표", "goal"),
                        ("대상자", "audience"), ("수준", "level"),
                        ("기간", "duration"), ("주제", "topic"), ("제한사항", "constraints")]:
        print(f"- {label}: {requirements[key]}")

    total = sum(type_counts.values())
    print(f"\n[진단 결과 요약] 총 {total}명")
    for type_name, count in type_counts.items():
        print(f"  - {type_name}: {count}명")

    print("\n[그룹 구성]")
    for group in groups.values():
        print(f"  - {group['name']} ({' · '.join(group['types'])}): {group['count']}명")


def print_curriculum(curriculum: dict):
    print("\n" + "=" * 60)
    print(f"과정명: {curriculum['program_title']}")
    print(f"대상 요약: {curriculum['target_summary']}")

    print("\n[공통 이론 수업]")
    for i, s in enumerate(curriculum["theory_sessions"], 1):
        print(f"\n  {i}. {s['title']}")
        for g in s["goals"]:
            print(f"     목표: {g}")
        for a in s["activities"]:
            print(f"     활동: {a}")

    print("\n[그룹별 맞춤 실습]")
    for group in curriculum["group_sessions"]:
        print(f"\n  ┌ {group['group_name']} | {group['target_types']} | {group['participant_count']}명")
        print(f"  │ 실습 포커스: {group['focus_description']}")
        for i, s in enumerate(group["sessions"], 1):
            print(f"\n  {i}. {s['title']}")
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

    vectorstore = setup_vector_store()
    chain = build_chain(vectorstore)

    requirements = collect_company_requirements()
    type_counts  = collect_type_counts()
    groups       = calculate_groups(type_counts)

    print_requirements_summary(requirements, type_counts, groups)

    print("\n[생성 중] 커리큘럼을 생성하고 있습니다...\n")
    result: CurriculumPlan = chain.invoke({
        "requirements": requirements,
        "groups": groups,
    })

    print_curriculum(result.model_dump())


if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as error:
        print(f"\n[오류] {error}")
