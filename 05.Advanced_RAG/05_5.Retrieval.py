import json
import logging
import os
import re
from time import perf_counter
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from schemas import CollectedInfo, CurriculumPlan, Message


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("curriculum_backend.retrieval")

HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3

COLLECTION_SYSTEM_PROMPT = dedent(
    """
    당신은 기업 AI 교육 커리큘럼 설계를 위한 정보 수집 어시스턴트다.
    아래 항목을 자연스럽고 대화형으로 한 번에 하나씩 수집해라.
    사용자가 여러 정보를 한 번에 말하면 이미 채워진 항목은 다시 묻지 마라.
    모든 항목이 수집되면 내용을 요약하고 마지막에 반드시 "[정보 수집 완료]"를 출력해라.

    수집 항목:
    - 회사명 또는 팀 이름
    - 교육 목표
    - 교육 대상자
    - 현재 AI 활용 수준 (입문/초급/중급)
    - 총 교육 기간 (일수)
    - 하루 교육 시간
    - 다루고 싶은 주제
    - 반영해야 할 조건 또는 제한사항
    - AX Compass 진단 결과 6개 유형별 인원
    """
).strip()

GENERATION_SYSTEM_PROMPT = dedent(
    """
    당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
    수집된 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계해라.

    출력 구조:
    - theory_sessions: 모든 참가자가 함께 듣는 공통 이론 세션. 4개 이상 6개 이하.
    - group_sessions: 3개 그룹이 각각 다르게 진행하는 실습 세션. 각 그룹당 2개 이상 3개 이하.

    규칙:
    1. 그룹별 실습은 해당 유형의 강점을 활용하고 보완 방향을 반영해 설계한다.
    2. 각 세션은 title, duration_hours, goals, activities를 포함한다.
    3. theory_sessions 전체 시간 + group_sessions 한 그룹 기준 전체 시간 = 총 교육 시간이어야 한다.
    4. 기업 교육답게 업무 적용 중심으로 구성한다.
    5. notes에는 유형별 특성과 기업 제약사항을 반영한 주의사항을 적는다.
    6. 참고 자료가 주어져도 내용을 그대로 복사하지 말고 현재 기업 상황에 맞게 재구성한다.
    """
).strip()

_MSG_CLS = {"user": HumanMessage, "assistant": AIMessage}


def to_lc_messages(messages: list[Message]) -> list:
    # FastAPI 요청 형식의 메시지를 LangChain 메시지 객체로 바꾼다.
    return [_MSG_CLS[message.role](content=message.content) for message in messages]


def _tokenize_for_bm25(text: str) -> list[str]:
    # 한국어/영문 키워드를 단순 토큰으로 잘라 BM25에 사용한다.
    return [token.lower() for token in re.findall(r"[0-9A-Za-z가-힣+#/.]{2,}", text or "")]


def _display_text(doc: Document) -> str:
    # 최종적으로 보여 주거나 LLM에 넣을 텍스트를 꺼낸다.
    return str(doc.metadata.get("display_text") or doc.page_content)


def _bm25_text(doc: Document) -> str:
    # BM25 키워드 검색에 쓸 전용 텍스트를 꺼낸다.
    return str(doc.metadata.get("bm25_text") or doc.page_content)


def _as_display_doc(doc: Document) -> Document:
    # 검색용 문서를 "보여 주는 텍스트 기준" 문서로 바꿔 결과 사용을 단순하게 만든다.
    return Document(page_content=_display_text(doc), metadata=doc.metadata)


def _doc_identity(doc: Document) -> str:
    # 같은 문서를 벡터 검색과 BM25가 각각 찾더라도 하나로 합칠 수 있게 고유 키를 만든다.
    metadata = doc.metadata or {}
    return str(
        metadata.get("chunk_id")
        or metadata.get("content_hash")
        or metadata.get("base_content_hash")
        or _tokenize_for_bm25(doc.page_content[:200])
    )


def _search_filter_key(search_filter: dict) -> str:
    # dict 형태의 필터를 캐시 키로 쓰기 위해 문자열로 바꾼다.
    return json.dumps(search_filter, sort_keys=True, ensure_ascii=False)


def _load_filtered_docs(
    vectorstore: Chroma,
    search_filter: dict,
    cache: dict[str, list[Document]],
) -> list[Document]:
    # BM25는 필터에 맞는 후보 문서를 먼저 모은 뒤 그 안에서 점수를 계산한다.
    cache_key = _search_filter_key(search_filter)
    if cache_key in cache:
        return cache[cache_key]

    result = vectorstore._collection.get(
        where=search_filter,
        include=["documents", "metadatas"],
    )
    docs = [
        Document(
            page_content=str((metadata or {}).get("display_text") or document),
            metadata=metadata or {},
        )
        for document, metadata in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    cache[cache_key] = docs
    return docs


def _bm25_search(candidate_docs: list[Document], query: str, *, k: int) -> list[Document]:
    # 필터에 맞는 후보 문서들 안에서 BM25 점수로 상위 문서를 고른다.
    if not candidate_docs:
        return []

    tokenized_query = _tokenize_for_bm25(query)
    if not tokenized_query:
        return candidate_docs[:k]

    tokenized_corpus = [_tokenize_for_bm25(_bm25_text(doc)) for doc in candidate_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    ranked_pairs = sorted(zip(candidate_docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in ranked_pairs[:k] if score > 0]


def _hybrid_fuse(dense_docs: list[Document], sparse_docs: list[Document], *, k: int) -> list[Document]:
    # 벡터 검색과 키워드 검색을 섞어 의미 유사도와 정확 키워드 매칭을 함께 반영한다.
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = _doc_identity(doc)
        doc_lookup[doc_id] = _as_display_doc(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + HYBRID_VECTOR_WEIGHT * (1 / rank)

    for rank, doc in enumerate(sparse_docs, start=1):
        doc_id = _doc_identity(doc)
        doc_lookup[doc_id] = _as_display_doc(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + HYBRID_BM25_WEIGHT * (1 / rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_lookup[doc_id] for doc_id in ranked_ids[:k]]


def _retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    k: int,
    search_filter: dict,
    label: str,
    corpus_cache: dict[str, list[Document]],
):
    # 실제 하이브리드 검색 진입점이다.
    # 1) 벡터 검색 2) BM25 검색 3) 둘을 합치기 순서로 동작한다.
    logger.info("[GENERATE][%s] retrieval start k=%s query=%s", label, k, query[:160])
    started_at = perf_counter()

    dense_k = max(k * 2, 6)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": dense_k,
            "filter": search_filter,
        },
    )
    dense_docs = [_as_display_doc(doc) for doc in retriever.invoke(query)]

    candidate_docs = _load_filtered_docs(vectorstore, search_filter, corpus_cache)
    sparse_docs = _bm25_search(candidate_docs, query, k=dense_k)
    docs = _hybrid_fuse(dense_docs, sparse_docs, k=k)

    logger.info(
        "[GENERATE][%s] retrieval done docs=%s dense=%s sparse=%s chars=%s elapsed=%.2fs",
        label,
        len(docs),
        len(dense_docs),
        len(sparse_docs),
        sum(len(doc.page_content) for doc in docs),
        perf_counter() - started_at,
    )
    return docs


def retrieve_group_context(
    vectorstore: Chroma,
    type_names: list[str],
    *,
    group_label: str,
    corpus_cache: dict[str, list[Document]],
) -> str:
    # AX Compass 문서 중에서 해당 그룹 유형과 맞는 설명만 찾아 하나의 참고문으로 합친다.
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    docs = _retrieve(
        vectorstore,
        query,
        k=max(4, len(type_names) * 2),
        search_filter={
            "$and": [
                {"doc_type": {"$eq": "ax_compass"}},
                {"type_name": {"$in": type_names}},
            ]
        },
        label=f"{group_label}_ax",
        corpus_cache=corpus_cache,
    )
    return "\n\n".join(doc.page_content for doc in docs)


def _keyword_tokens(*texts: str) -> list[str]:
    # 사용자 요구사항에서 핵심 키워드만 뽑아 간단한 규칙 기반 점수 계산에 쓴다.
    tokens: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in re.findall(r"[0-9A-Za-z가-힣+#/.]{2,}", text or ""):
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            tokens.append(token)
    return tokens


def _build_curriculum_query(info: CollectedInfo) -> str:
    # 커리큘럼 예시 검색에 쓸 질의를 사용자 요구사항 중심으로 만든다.
    return dedent(
        f"""
        기업 맞춤형 AI 교육 커리큘럼 예시를 찾는다.
        강의 주제: {info.topic}
        교육 목표: {info.goal}
        교육 대상자: {info.audience}
        현재 수준: {info.level}
        반영 조건 및 제한사항: {info.constraints}
        총 교육 시간: {info.days * info.hours_per_day}시간
        실무 적용 중심의 기업 교육 사례
        """
    ).strip()


def _score_curriculum_doc(doc: Document, info: CollectedInfo) -> int:
    # 검색된 커리큘럼 예시 중에서 현재 기업 요구사항과 더 잘 맞는 문서를 다시 고른다.
    haystack = " ".join(
        [
            _bm25_text(doc),
            str(doc.metadata.get("course_name", "")),
        ]
    ).lower()
    primary_tokens = _keyword_tokens(info.topic, info.goal)
    secondary_tokens = _keyword_tokens(info.audience, info.level, info.constraints)

    score = 0
    for token in primary_tokens:
        if token.lower() in haystack:
            score += 3
    for token in secondary_tokens:
        if token.lower() in haystack:
            score += 1
    return score


def retrieve_curriculum_examples(
    vectorstore: Chroma,
    info: CollectedInfo,
    *,
    k: int = 2,
    corpus_cache: dict[str, list[Document]],
) -> str:
    # 커리큘럼 예시는 하이브리드 검색 후 한 번 더 재정렬해서 정말 참고할 문서만 남긴다.
    query = _build_curriculum_query(info)
    candidates = _retrieve(
        vectorstore,
        query,
        k=max(6, k * 3),
        search_filter={"doc_type": {"$eq": "curriculum_example"}},
        label="curriculum_examples",
        corpus_cache=corpus_cache,
    )
    ranked_docs = sorted(
        candidates,
        key=lambda doc: (_score_curriculum_doc(doc, info), len(doc.page_content)),
        reverse=True,
    )
    selected_docs = ranked_docs[:k]
    logger.info(
        "[GENERATE][curriculum_examples] rerank selected=%s candidate_scores=%s",
        len(selected_docs),
        [_score_curriculum_doc(doc, info) for doc in selected_docs],
    )
    return "\n\n---\n\n".join(doc.page_content for doc in selected_docs)


def build_chain(vectorstore: Chroma):
    # 검색 결과를 모아 최종 커리큘럼 생성 LLM으로 넘기는 체인을 만든다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(CurriculumPlan)
    corpus_cache: dict[str, list[Document]] = {}

    def retrieve_and_build_messages(input_dict: dict) -> list:
        # generate 요청 하나를 처리할 때 필요한 참고 자료를 모두 모아 마지막 프롬프트를 만든다.
        started_at = perf_counter()
        conversation = input_dict["conversation"]
        info: CollectedInfo = input_dict["info"]
        groups = input_dict["groups"]

        logger.info(
            "[GENERATE] build messages start company=%s topic=%s level=%s conversation_messages=%s",
            info.company_name,
            info.topic,
            info.level,
            len(conversation),
        )

        total_hours = info.days * info.hours_per_day
        theory_hours = round(total_hours * 0.65)
        group_hours = total_hours - theory_hours

        ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]
        ctx_a = retrieve_group_context(vectorstore, ga["types"], group_label="group_a", corpus_cache=corpus_cache)
        ctx_b = retrieve_group_context(vectorstore, gb["types"], group_label="group_b", corpus_cache=corpus_cache)
        ctx_c = retrieve_group_context(vectorstore, gc["types"], group_label="group_c", corpus_cache=corpus_cache)

        curriculum_examples = retrieve_curriculum_examples(vectorstore, info, corpus_cache=corpus_cache)

        chat_history = [message for message in conversation if not isinstance(message, SystemMessage)]

        rag_content = dedent(
            f"""
            위 대화에서 수집한 요구사항을 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

            [시간 배분 기준]
            총 교육 시간: {total_hours}시간
            - 공통 이론(theory_sessions) 전체 합계: 정확히 {theory_hours}시간
            - 그룹 실습(group_sessions) 한 그룹 기준 전체 합계: 정확히 {group_hours}시간
            - group_sessions는 3개 그룹이 동시에 진행되므로 각 그룹의 duration_hours 합계는 동일해야 한다.

            [그룹 구성]
            - {ga['name']} ({' / '.join(ga['types'])}): {ga['count']}명
            - {gb['name']} ({' / '.join(gb['types'])}): {gb['count']}명
            - {gc['name']} ({' / '.join(gc['types'])}): {gc['count']}명

            [AX Compass 참고 자료]
            === 그룹 A ({' / '.join(ga['types'])}) ===
            {ctx_a}

            === 그룹 B ({' / '.join(gb['types'])}) ===
            {ctx_b}

            === 그룹 C ({' / '.join(gc['types'])}) ===
            {ctx_c}

            [커리큘럼 예시 참고 자료]
            {curriculum_examples}
            """
        ).strip()
        logger.info(
            "[GENERATE] build messages done rag_chars=%s chat_history=%s elapsed=%.2fs",
            len(rag_content),
            len(chat_history),
            perf_counter() - started_at,
        )

        return [SystemMessage(content=GENERATION_SYSTEM_PROMPT)] + chat_history + [HumanMessage(content=rag_content)]

    return RunnableLambda(retrieve_and_build_messages) | structured_llm
