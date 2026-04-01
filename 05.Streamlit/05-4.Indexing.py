import hashlib
import os
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 이 파일은 "문서를 읽고 -> 정리하고 -> 잘게 나누고 -> 벡터 DB에 저장"하는 역할만 담당한다.
BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
PDF_PATH = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")

AX_COLLECTION_NAME = "ax_compass_profiles_v4_contextual"
CURRICULUM_COLLECTION_NAME = "curriculum_examples_v4_contextual"

AX_CHUNK_SIZE = 900
AX_CHUNK_OVERLAP = 120
CURRICULUM_CHUNK_SIZE = 700
CURRICULUM_CHUNK_OVERLAP = 80

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


def _clean_text(text: str) -> str:
    # PDF/Excel에서 나온 들쭉날쭉한 공백을 정리해 검색 품질을 조금 더 안정적으로 만든다.
    normalized_lines = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif normalized_lines and not previous_blank:
            normalized_lines.append("")
            previous_blank = True
    return "\n".join(normalized_lines).strip()


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _page_number_from_metadata(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return None


def _extract_section_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:120]
    return fallback


def _tag_ax_type(metadata: dict[str, Any], text: str) -> None:
    # 본문에 유형명이 보이면 메타데이터에도 같이 넣어, 이후 필터 검색에 활용한다.
    for type_name, info in TYPE_INFO.items():
        if type_name in text:
            metadata.update(
                {
                    "type_name": type_name,
                    "group": info["group"],
                    "english": info["english"],
                }
            )
            return


def _infer_curriculum_content_type(text: str) -> str:
    # 커리큘럼 문서가 "목표 / 활동 / 세션 계획" 중 무엇에 가까운지 대략 분류한다.
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("activity", "activities", "exercise", "workshop", "practice")):
        return "activity_plan"
    if any(keyword in lowered for keyword in ("goal", "goals", "objective", "learning outcome")):
        return "learning_goal"
    if any(keyword in lowered for keyword in ("case", "template", "agenda", "session")):
        return "session_plan"
    return "curriculum_reference"


def _format_excel_content(text: str) -> str:
    # Excel은 셀 구조가 중요해서, 단순 문자열보다 태그가 붙은 형태로 바꿔 두는 편이 읽기 쉽다.
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in raw_line:
            cells = [cell.strip() for cell in raw_line.split("\t") if cell.strip()]
            if cells:
                rows.append("[row] " + " | ".join(cells))
                continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() and value.strip():
                rows.append(f"[field] {key.strip()}: {value.strip()}")
                continue
        rows.append(f"[entry] {' '.join(line.split())}")
    return "\n".join(rows).strip()


def _build_structured_document(content: str, metadata: dict[str, Any]) -> Document:
    # 청킹 전에 핵심 메타데이터를 본문 앞에 붙여 둔다.
    # 이렇게 하면 잘린 청크도 "어느 문서의 어떤 내용인지" 문맥을 더 잘 유지한다.
    header_fields = [
        "doc_type",
        "source_name",
        "course_name",
        "sheet_name",
        "page_number",
        "section_title",
        "module_name",
        "content_type",
        "type_name",
        "group",
        "english",
    ]
    header_lines = []
    for field_name in header_fields:
        value = metadata.get(field_name)
        if value not in (None, ""):
            header_lines.append(f"[{field_name}] {value}")
    header_lines.append("[content]")
    header_lines.append(content)
    return Document(page_content="\n".join(header_lines).strip(), metadata=metadata)


def _annotate_chunks(chunks: list[Document]) -> list[Document]:
    # 각 청크에 추적용 메타데이터를 붙여 둔다.
    # chunk_id / content_hash는 중복 방지나 이후 증분 인덱싱의 기반이 된다.
    source_counters: dict[tuple[str, str, str], int] = {}
    for chunk in chunks:
        source_key = (
            str(chunk.metadata.get("source_file", "")),
            str(chunk.metadata.get("page_number", chunk.metadata.get("sheet_name", ""))),
            str(chunk.metadata.get("section_title", "")),
        )
        chunk_index = source_counters.get(source_key, 0)
        source_counters[source_key] = chunk_index + 1

        content_hash = _short_hash(chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_id": f"{chunk.metadata.get('source_name', 'doc')}:{chunk_index}:{content_hash}",
                "content_hash": content_hash,
                "word_count": len(chunk.page_content.split()),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if chunk.metadata.get("doc_type") == "ax_compass":
            _tag_ax_type(chunk.metadata, chunk.page_content)
    return chunks


CONTEXTUAL_PROMPT = dedent(
    """
    <document>
    {document}
    </document>
    다음은 위 문서에서 추출한 청크입니다. 검색 성능을 높이기 위해 이 청크를 전체 문서 맥락 안에서 간결하게 설명하라. 설명만 출력해라.
    <chunk>
    {chunk}
    </chunk>
    """
).strip()


def _doc_key(metadata: dict) -> str:
    # 청크 → 원본 문서를 찾는 데 쓰는 키. page_number 우선, 없으면 sheet_name.
    return "{}|{}".format(
        metadata.get("source_file", ""),
        metadata.get("page_number") or metadata.get("sheet_name") or "",
    )


def _apply_contextual_retrieval(
    source_docs: list[Document],
    chunks: list[Document],
    llm: ChatOpenAI,
) -> list[Document]:
    # 각 청크에 대해 원본 문서 전체를 참고해 맥락 요약을 생성하고, [context] 태그로 앞에 붙인다.
    # 청크가 잘려도 "이 청크가 전체 문서의 어느 부분인지" 정보가 임베딩에 함께 반영된다.
    doc_map = {_doc_key(doc.metadata): doc.page_content for doc in source_docs}
    enriched: list[Document] = []
    for chunk in chunks:
        full_text = doc_map.get(_doc_key(chunk.metadata), "")
        if full_text:
            try:
                response = llm.invoke(
                    CONTEXTUAL_PROMPT.format(document=full_text, chunk=chunk.page_content)
                )
                context = response.content.strip()
                new_content = f"[context] {context}\n{chunk.page_content}"
                new_hash = _short_hash(new_content)
                chunk = Document(
                    page_content=new_content,
                    metadata={
                        **chunk.metadata,
                        "content_hash": new_hash,
                        "chunk_id": "{}:{}:{}".format(
                            chunk.metadata.get("source_name", "doc"),
                            chunk.metadata.get("chunk_index", 0),
                            new_hash,
                        ),
                        "contextual": True,
                    },
                )
            except Exception as exc:
                print(f"[ContextualRetrieval] 맥락 생성 실패: {exc}")
        enriched.append(chunk)
    return enriched


def _split_documents_with_strategy(
    docs: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    # 문서 종류마다 다른 크기로 자르되, [content] 경계를 먼저 보도록 분리 기준을 준다.
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n[content]", "\n\n", "\n", " "],
    )
    return _annotate_chunks(splitter.split_documents(docs))


def _load_pdf_pages(fpath: str, metadata_builder) -> list[Document]:
    # PDF 공통 로더. "페이지 읽기 + 텍스트 정리 + 메타데이터 생성"을 한 번에 처리한다.
    documents = []
    for page in PyPDFLoader(fpath).load():
        body = _clean_text(page.page_content)
        if not body:
            continue
        metadata = metadata_builder(page, body, fpath)
        documents.append(_build_structured_document(body, metadata))
    return documents


def load_ax_compass_documents() -> list[Document]:
    # AX Compass는 유형 설명 문서라서, page_number / section_title / type_name 정보가 특히 중요하다.
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")

    print("[RAG] AXCompass PDF 로드 중...")

    def _build_meta(page, body, fpath):
        source_file = page.metadata.get("source", fpath)
        page_number = _page_number_from_metadata(page.metadata.get("page"))
        section_title = _extract_section_title(body, f"ax_page_{page_number or 'unknown'}")
        metadata = {
            "doc_type": "ax_compass",
            "source_file": source_file,
            "source_name": os.path.basename(source_file),
            "page_number": page_number,
            "section_title": section_title,
            "content_type": "reference_profile",
        }
        _tag_ax_type(metadata, body)
        return metadata

    return _load_pdf_pages(PDF_PATH, _build_meta)


def load_curriculum_pdf_documents() -> list[Document]:
    # 커리큘럼 PDF 예시는 과정명과 섹션명을 같이 남겨 두면 나중에 참고 사례 검색에 유리하다.
    documents = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".pdf") or fname == "AXCompass.pdf":
            continue

        fpath = os.path.join(DATA_DIR, fname)
        course_name = os.path.splitext(fname)[0]
        print(f"[RAG] 커리큘럼 PDF 로드: {fname}")

        def _build_meta(page, body, pdf_path, _course=course_name):
            source_file = page.metadata.get("source", pdf_path)
            page_number = _page_number_from_metadata(page.metadata.get("page"))
            section_title = _extract_section_title(body, _course)
            return {
                "doc_type": "curriculum_example",
                "source_file": source_file,
                "source_name": os.path.basename(source_file),
                "course_name": _course,
                "page_number": page_number,
                "section_title": section_title,
                "module_name": section_title,
                "content_type": _infer_curriculum_content_type(body),
            }

        documents.extend(_load_pdf_pages(fpath, _build_meta))

    return documents


def load_curriculum_excel_documents() -> list[Document]:
    # Excel 예시는 시트 구조를 최대한 살려야 검색 시 의미가 덜 깨진다.
    documents = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".xlsx"):
            continue

        fpath = os.path.join(DATA_DIR, fname)
        course_name = os.path.splitext(fname)[0]
        print(f"[RAG] 커리큘럼 Excel 로드: {fname}")

        for sheet_index, doc in enumerate(UnstructuredExcelLoader(fpath).load(), start=1):
            body = _format_excel_content(doc.page_content)
            if not body:
                continue

            source_file = doc.metadata.get("source", fpath)
            sheet_name = (
                doc.metadata.get("page_name")
                or doc.metadata.get("sheet_name")
                or f"sheet_{sheet_index}"
            )
            section_title = _extract_section_title(body, f"{course_name}_{sheet_name}")
            metadata = {
                "doc_type": "curriculum_example",
                "source_file": source_file,
                "source_name": os.path.basename(source_file),
                "course_name": course_name,
                "sheet_name": sheet_name,
                "section_title": section_title,
                "module_name": section_title,
                "content_type": _infer_curriculum_content_type(body),
            }
            documents.append(_build_structured_document(body, metadata))

    return documents


def load_ax_compass_chunks(contextual: bool = False) -> list[Document]:
    # AX Compass는 설명형 문서라 청크를 조금 더 크게 잡는다.
    docs = load_ax_compass_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=AX_CHUNK_SIZE,
        chunk_overlap=AX_CHUNK_OVERLAP,
    )
    if contextual:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        print(f"[ContextualRetrieval] AX Compass 맥락 생성 중... ({len(chunks)}개 청크)")
        chunks = _apply_contextual_retrieval(docs, chunks, llm)
    print(f"[RAG] AX Compass 청크 수: {len(chunks)}")
    return chunks


def load_curriculum_chunks(contextual: bool = False) -> list[Document]:
    # 커리큘럼 예시는 세션 단위 정보가 많아서 AX Compass보다 약간 더 촘촘하게 자른다.
    docs = load_curriculum_pdf_documents() + load_curriculum_excel_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=CURRICULUM_CHUNK_SIZE,
        chunk_overlap=CURRICULUM_CHUNK_OVERLAP,
    )
    if contextual:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        print(f"[ContextualRetrieval] 커리큘럼 맥락 생성 중... ({len(chunks)}개 청크)")
        chunks = _apply_contextual_retrieval(docs, chunks, llm)
    print(f"[RAG] 커리큘럼 예시 청크 수: {len(chunks)}")
    return chunks


def _collection_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


def _create_vector_store(embeddings: OpenAIEmbeddings, collection_name: str) -> Chroma:
    # 컬렉션 이름을 분리해 두면 AX Compass와 커리큘럼 예시를 독립적으로 다룰 수 있다.
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )


def _get_indexed_hashes(vectorstore: Chroma) -> set[str]:
    # 이미 들어간 청크의 해시를 읽어 와서, 같은 내용이 다시 들어가지 않도록 한다.
    try:
        result = vectorstore._collection.get(include=["metadatas"])
    except Exception:
        return set()

    return {
        metadata.get("content_hash", "")
        for metadata in result.get("metadatas", [])
        if metadata and metadata.get("content_hash")
    }


def _filter_new_documents(docs: list[Document], indexed_hashes: set[str]) -> list[Document]:
    return [doc for doc in docs if doc.metadata.get("content_hash", "") not in indexed_hashes]


def _ensure_index(vectorstore: Chroma, loader, label: str) -> None:
    # 첫 실행이면 전체 인덱싱, 이후에는 새 해시만 추가한다.
    # 아직 "수정/삭제 동기화"까지는 아니지만, 완전 재색인보다 훨씬 가볍게 시작할 수 있다.
    existing_count = _collection_count(vectorstore)
    chunks = loader()

    if existing_count == 0:
        print(f"[VectorDB] {label} 컬렉션 인덱싱 중...")
        if chunks:
            vectorstore.add_documents(chunks)
        print(f"[VectorDB] {label} 컬렉션 완료 ({len(chunks)}개 청크)")
        return

    indexed_hashes = _get_indexed_hashes(vectorstore)
    new_chunks = _filter_new_documents(chunks, indexed_hashes)
    if new_chunks:
        print(f"[VectorDB] {label} 신규 청크 {len(new_chunks)}개 추가 중...")
        vectorstore.add_documents(new_chunks)
        print(f"[VectorDB] {label} 추가 완료 ({_collection_count(vectorstore)}개 청크)")
    else:
        print(f"[VectorDB] {label} 변경 없음. 기존 컬렉션 유지 ({existing_count}개 청크)")


def setup_vector_stores(contextual: bool = False) -> dict[str, Chroma]:
    # 문서 성격이 다른 두 컬렉션을 분리해 두면 검색 의도를 더 명확하게 나눌 수 있다.
    # contextual=True 이면 각 청크 앞에 LLM이 생성한 맥락 요약이 붙어 검색 정확도가 올라간다.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    ax_store = _create_vector_store(embeddings, AX_COLLECTION_NAME)
    curriculum_store = _create_vector_store(embeddings, CURRICULUM_COLLECTION_NAME)

    _ensure_index(ax_store, lambda: load_ax_compass_chunks(contextual=contextual), "AX Compass")
    _ensure_index(curriculum_store, lambda: load_curriculum_chunks(contextual=contextual), "커리큘럼 예시")

    return {
        "ax_compass": ax_store,
        "curriculum_examples": curriculum_store,
    }


def setup_vector_store(contextual: bool = False):
    return setup_vector_stores(contextual=contextual)
