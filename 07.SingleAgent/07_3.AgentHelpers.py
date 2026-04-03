import json
import logging
import os
import re
from textwrap import dedent
from time import perf_counter

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

# 05.Advanced_RAG의 Indexing 모듈을 재사용한다.
# vectorDB와 Data 경로는 환경변수(APP_BASE_DIR)로 제어한다.
import importlib.util
import sys


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("single_agent.helpers")


# ---------------------------------------------------------------------------
# VectorStore 초기화 (05.Advanced_RAG의 setup_vector_store 재사용)
# ---------------------------------------------------------------------------

def _load_indexing_module():
    # Docker 안에서는 이 파일이 /app 바로 아래에 복사되므로, 프로젝트 루트는
    # APP_BASE_DIR(/app) 기준으로 잡고 로컬에서는 현재 파일의 상위 폴더를 사용한다.
    base = os.getenv("APP_BASE_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "05.Advanced_RAG", "05_4.Indexing.py")
    spec = importlib.util.spec_from_file_location("indexing_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["indexing_module"] = module
    spec.loader.exec_module(module)
    return module


def setup_vector_store() -> Chroma:
    indexing = _load_indexing_module()
    return indexing.setup_vector_store()


# ---------------------------------------------------------------------------
# Hybrid Search 공통 유틸
# ---------------------------------------------------------------------------

HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3
USE_RERANKING = True
RERANK_MODEL = ""

_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Ranker(model_name=RERANK_MODEL) if RERANK_MODEL else Ranker()
    return _RERANKER


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[0-9A-Za-z가-힣+#/.]{2,}", text or "")]


def _doc_id(doc: Document) -> str:
    m = doc.metadata or {}
    return str(
        m.get("chunk_id") or m.get("content_hash") or m.get("base_content_hash")
        or _tokenize(doc.page_content[:200])
    )


def _display_text(doc: Document) -> str:
    return str(doc.metadata.get("display_text") or doc.page_content)


def _bm25_text(doc: Document) -> str:
    return str(doc.metadata.get("bm25_text") or doc.page_content)


def _as_display(doc: Document) -> Document:
    return Document(page_content=_display_text(doc), metadata=doc.metadata)


def _hybrid_fuse(dense: list[Document], sparse: list[Document], *, k: int) -> list[Document]:
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}
    for rank, doc in enumerate(dense, 1):
        did = _doc_id(doc)
        lookup[did] = _as_display(doc)
        scores[did] = scores.get(did, 0.0) + HYBRID_VECTOR_WEIGHT * (1 / rank)
    for rank, doc in enumerate(sparse, 1):
        did = _doc_id(doc)
        lookup[did] = _as_display(doc)
        scores[did] = scores.get(did, 0.0) + HYBRID_BM25_WEIGHT * (1 / rank)
    return [lookup[did] for did in sorted(scores, key=scores.get, reverse=True)[:k]]


def _rerank(query: str, docs: list[Document], *, k: int, label: str) -> list[Document]:
    if not USE_RERANKING or not docs:
        return docs[:k]
    passages = [{"id": _doc_id(d), "text": _display_text(d), "meta": d.metadata} for d in docs]
    try:
        reranked = _get_reranker().rerank(RerankRequest(query=query, passages=passages))
        result = [Document(page_content=p["text"], metadata=p.get("meta", {})) for p in reranked[:k]]
        logger.info("[%s] rerank done input=%s output=%s", label, len(docs), len(result))
        return result
    except Exception as err:
        logger.warning("[%s] rerank skipped: %s", label, err)
        return docs[:k]


_corpus_cache: dict[str, tuple[list[Document], BM25Okapi | None]] = {}


def _load_filtered_docs(
    vectorstore: Chroma,
    search_filter: dict,
) -> tuple[list[Document], BM25Okapi | None]:
    cache_key = json.dumps(search_filter, sort_keys=True, ensure_ascii=False)
    if cache_key in _corpus_cache:
        return _corpus_cache[cache_key]

    result = vectorstore._collection.get(where=search_filter, include=["documents", "metadatas"])
    docs = [
        Document(
            page_content=str((m or {}).get("display_text") or d),
            metadata=m or {},
        )
        for d, m in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    tokenized = [_tokenize(_bm25_text(doc)) for doc in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    _corpus_cache[cache_key] = (docs, bm25)
    return _corpus_cache[cache_key]


def _bm25_search(docs: list[Document], bm25: BM25Okapi | None, query: str, *, k: int) -> list[Document]:
    if not docs or bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return docs[:k]
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:k] if score > 0]


def _retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    k: int,
    search_filter: dict,
    label: str,
) -> list[Document]:
    logger.info("[%s] retrieval start k=%s query=%s", label, k, query[:120])
    t0 = perf_counter()

    dense_k = max(k * 2, 6)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": dense_k, "filter": search_filter},
    )
    dense = [_as_display(d) for d in retriever.invoke(query)]
    candidate_docs, bm25 = _load_filtered_docs(vectorstore, search_filter)
    sparse = _bm25_search(candidate_docs, bm25, query, k=dense_k)
    hybrid = _hybrid_fuse(dense, sparse, k=dense_k)
    docs = _rerank(query, hybrid, k=k, label=label)

    logger.info(
        "[%s] retrieval done docs=%s elapsed=%.2fs",
        label, len(docs), perf_counter() - t0,
    )
    return docs


# ---------------------------------------------------------------------------
# 도구별 검색 함수
# ---------------------------------------------------------------------------

def retrieve_ax_compass(vectorstore: Chroma, query: str, type_names: list[str] | None = None) -> str:
    """AX Compass 문서에서 유형별 특성 및 교육 접근법을 검색한다."""
    if type_names:
        search_filter = {
            "$and": [
                {"doc_type": {"$eq": "ax_compass"}},
                {"type_name": {"$in": type_names}},
            ]
        }
        label = f"ax_compass({'|'.join(type_names)})"
        k = max(4, len(type_names) * 2)
    else:
        search_filter = {"doc_type": {"$eq": "ax_compass"}}
        label = "ax_compass"
        k = 4

    docs = _retrieve(vectorstore, query, k=k, search_filter=search_filter, label=label)
    if not docs:
        return "관련 AX Compass 자료를 찾지 못했습니다."
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_curriculum_examples(vectorstore: Chroma, query: str, k: int = 3) -> str:
    """커리큘럼 예시 문서에서 유사한 교육 사례를 검색한다."""
    docs = _retrieve(
        vectorstore,
        query,
        k=k,
        search_filter={"doc_type": {"$eq": "curriculum_example"}},
        label="curriculum_examples",
    )
    if not docs:
        return "관련 커리큘럼 예시를 찾지 못했습니다."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Tavily 웹 검색
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Tavily API를 이용해 최신 웹 정보를 검색한다."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "TAVILY_API_KEY가 설정되지 않아 웹 검색을 사용할 수 없습니다."

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as err:
        logger.warning("[web_search] 검색 실패: %s", err)
        return f"웹 검색 중 오류가 발생했습니다: {err}"

    parts: list[str] = []
    if data.get("answer"):
        parts.append(f"[요약]\n{data['answer']}")

    for r in data.get("results", []):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"**{title}**\n{url}\n{content}")

    return "\n\n---\n\n".join(parts) if parts else "검색 결과가 없습니다."


# ---------------------------------------------------------------------------
# 커리큘럼 생성 (OpenAI structured output via with_structured_output)
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = dedent(
    """
    당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
    수집된 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 하루 단위로 설계해라.

    출력 구조 (반드시 준수):
    - groups: 실제 인원이 있는 그룹만 생성한다. (group_name, target_types, participant_count, focus_description)
    - daily_schedules: 교육 기간 N일 각각에 대해 DaySchedule 1개씩 반드시 생성한다.
      - day: 1부터 시작하는 일차 번호
      - theme: 해당 일의 핵심 주제 (한 줄)
      - common_sessions: 모든 참여자가 함께 수강하는 공통 세션 목록
      - group_sessions: 실제 인원이 있는 그룹만 생성. 그룹별 세션이 있는 일차에는 모든 그룹이 병렬 참여.
        - group_name: 그룹 이름
        - sessions: 해당 그룹의 세션 목록

    세션 필드 (모든 세션에 반드시 포함):
    - title: 실제 강의 제목처럼 구체적인 세션명
    - session_type: "공통 이론" / "공통 실습" / "그룹별 프로젝트" / "그룹별 심화 적용" 중 하나
    - target: 대상 ("전체" 또는 "그룹 A" 등 실제 그룹명)
    - duration_hours: 진행 시간 (소수점 0.5 단위)
    - purpose: 이 세션을 편성한 목적 (1~2문장)
    - goals: 학습목표 — 수업 후 학습자가 설명·판단·수행·적용할 수 있는 행동 중심으로 3개 이상
    - contents: 학습내용 — 무엇을 어떤 순서로 다루고 어떤 예시·활동이 포함되는지 3개 이상
    - method: 진행 방식 (강의, 실습, 토의, 발표, 팀 프로젝트 등 구체적으로)
    - expected_effect: 기대효과 — 업무 적용 가능성, 실무 역량 향상, 문제 해결력 강화 관점으로 서술

    시간 규칙:
    - 각 일차의 common_sessions 합계 + 한 그룹 sessions 합계 = hours_per_day
    - 그룹별 세션이 있는 일차에서 모든 그룹의 sessions 시간 합계는 동일해야 한다.
    - 반드시 days 수만큼의 DaySchedule을 생성한다.

    설계 규칙:
    1. 일차별 주제(theme)가 자연스럽게 이어지도록 전체 흐름을 먼저 구성한다.
    2. 공통 실습은 도구 기능 익히기·기본 예제 수준까지만 허용하고, 실무 적용·결과물 제작은 그룹별로만 편성한다.
    3. 그룹별 세션이 편성된 일차에는 실제 인원이 있는 모든 그룹이 같은 시간대에 병렬 참여한다.
    4. 그룹별 세션은 그룹마다 세션명·학습목표·학습내용·기대효과가 실질적으로 달라야 한다.
    5. 교육 대상자의 직무·업무 맥락에 맞는 예시·실습·산출물을 반영한다.
    6. notes에는 유형별 특성과 기업 제약사항을 반영한 주의사항을 적는다.
    7. 참고 자료는 참고만 하고 현재 기업 상황에 맞게 재구성한다.
    """
).strip()


def _scale_sessions(sessions: list, target_total: float) -> None:
    """세션 목록의 duration_hours 합계를 target_total에 맞게 비율 보정한다 (0.5시간 단위)."""
    if not sessions:
        return
    actual = sum(s.duration_hours for s in sessions)
    if actual <= 0:
        return
    ratio = target_total / actual
    for s in sessions:
        s.duration_hours = round(s.duration_hours * ratio * 2) / 2
    residual = round(target_total - sum(s.duration_hours for s in sessions), 1)
    if residual:
        sessions[-1].duration_hours = round(sessions[-1].duration_hours + residual, 1)


def _correct_hours_daily(result, hours_per_day: float):
    """
    하루 단위 커리큘럼의 시간 합계를 보정한다.
    각 일차의 common_sessions 합계 + 그룹 sessions 합계(한 그룹 기준) = hours_per_day
    """
    for day in result.daily_schedules:
        common_total = sum(s.duration_hours for s in day.common_sessions)
        # 그룹 sessions 시간은 모두 동일해야 하므로 첫 그룹 기준으로 계산
        group_total = (
            sum(s.duration_hours for s in day.group_sessions[0].sessions)
            if day.group_sessions else 0
        )
        actual_day = common_total + group_total

        if actual_day <= 0 or abs(actual_day - hours_per_day) <= 0.01:
            continue

        ratio = hours_per_day / actual_day
        target_common = round(common_total * ratio * 2) / 2
        target_group = round(hours_per_day - target_common, 1)

        logger.info(
            "[generate_curriculum] day %s hour correction: actual=%.1f target=%.1f "
            "common %.1f→%.1f group %.1f→%.1f",
            day.day, actual_day, hours_per_day,
            common_total, target_common, group_total, target_group,
        )

        _scale_sessions(day.common_sessions, target_common)
        for gs in day.group_sessions:
            _scale_sessions(gs.sessions, target_group)

    return result


def generate_curriculum(
    info_dict: dict,
    *,
    ax_context_a: str = "",
    ax_context_b: str = "",
    ax_context_c: str = "",
    curriculum_context: str = "",
    web_context: str = "",
) -> dict:
    """
    에이전트가 직접 검색해서 모은 컨텍스트를 받아 커리큘럼을 하루 단위로 생성한다.
    내부에서 RAG를 호출하지 않는다 — 검색은 에이전트가 도구를 통해 미리 수행해야 한다.
    """
    days = info_dict.get("days", 1)
    hours_per_day = info_dict.get("hours_per_day", 8)
    total_hours = days * hours_per_day

    groups = {
        "group_a": {
            "name": "그룹 A", "types": ["균형형", "이해형"],
            "count": info_dict.get("count_balanced", 0) + info_dict.get("count_learner", 0),
        },
        "group_b": {
            "name": "그룹 B", "types": ["과신형", "실행형"],
            "count": info_dict.get("count_overconfident", 0) + info_dict.get("count_doer", 0),
        },
        "group_c": {
            "name": "그룹 C", "types": ["판단형", "조심형"],
            "count": info_dict.get("count_analyst", 0) + info_dict.get("count_cautious", 0),
        },
    }
    ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]

    prompt = dedent(
        f"""
        수집된 요구사항과 에이전트가 검색한 참고 자료를 바탕으로 맞춤형 교육 커리큘럼을 하루 단위로 설계해줘.

        [기업 정보]
        회사명: {info_dict.get('company_name', '')}
        교육 목표: {info_dict.get('goal', '')}
        교육 대상: {info_dict.get('audience', '')}
        AI 활용 수준: {info_dict.get('level', '')}
        핵심 주제: {info_dict.get('topic', '')}
        제약 사항: {info_dict.get('constraints', '')}

        [시간 제약 — 반드시 준수]
        총 교육 기간: {days}일
        하루 교육 시간: {hours_per_day}시간 (총 {total_hours}시간)
        daily_schedules는 반드시 {days}개 (1일차 ~ {days}일차) 모두 생성한다.
        각 일차 조건: common_sessions 합계 + 그룹 sessions 합계(한 그룹 기준) = {hours_per_day}시간
        세 그룹의 sessions 시간 합계는 동일해야 한다 (내용은 달라도 됨).

        [그룹 구성]
        - {ga['name']} ({' / '.join(ga['types'])}): {ga['count']}명
        - {gb['name']} ({' / '.join(gb['types'])}): {gb['count']}명
        - {gc['name']} ({' / '.join(gc['types'])}): {gc['count']}명

        [그룹 A ({' / '.join(ga['types'])}) — AX Compass 유형 특성]
        아래 특성·강점·보완 방향을 그룹 A 실습 세션 설계에 반드시 반영할 것.
        {ax_context_a or "자료 없음"}

        [그룹 B ({' / '.join(gb['types'])}) — AX Compass 유형 특성]
        아래 특성·강점·보완 방향을 그룹 B 실습 세션 설계에 반드시 반영할 것.
        {ax_context_b or "자료 없음"}

        [그룹 C ({' / '.join(gc['types'])}) — AX Compass 유형 특성]
        아래 특성·강점·보완 방향을 그룹 C 실습 세션 설계에 반드시 반영할 것.
        {ax_context_c or "자료 없음"}

        [내부 커리큘럼 예시]
        요구사항과 일치하는 기존 사례가 있으면 일차 구성 방식을 참고하되,
        그대로 복사하지 말고 현재 기업 상황에 맞게 재구성한다.
        {curriculum_context or "일치하는 내부 예시 없음"}

        [최신 트렌드 및 실제 커리큘럼 사례 (웹 검색)]
        최신 동향과 실제 강의 구성 방식을 참고해 커리큘럼의 현실성을 높인다.
        {web_context or "자료 없음"}
        """
    ).strip()

    import importlib.util as _ilu, sys as _sys
    schemas_path = os.path.join(os.path.dirname(__file__), "07_2.AgentSchemas.py")
    # 스키마 모듈은 항상 최신 버전으로 재로드한다
    _spec = _ilu.spec_from_file_location("agent_schemas", schemas_path)
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["agent_schemas"] = _mod
    _spec.loader.exec_module(_mod)
    CurriculumPlan = _sys.modules["agent_schemas"].CurriculumPlan

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    result = llm.with_structured_output(CurriculumPlan).invoke(
        [SystemMessage(content=GENERATION_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    result = _correct_hours_daily(result, hours_per_day)
    return result.model_dump()
