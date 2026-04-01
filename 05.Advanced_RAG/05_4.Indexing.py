import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
PDF_PATH = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Contextual Retrieval을 켜면 청크 앞에 "이 청크가 문서 전체에서 어떤 의미인지" 요약을 붙인다.
# 컬렉션 이름을 분리해 두면 기존 인덱스와 섞이지 않는다.
USE_CONTEXTUAL_RETRIEVAL = _env_flag("USE_CONTEXTUAL_RETRIEVAL", default=True)
COLLECTION_NAME = "advanced_rag_v4_contextual" if USE_CONTEXTUAL_RETRIEVAL else "advanced_rag_v3"
CONTEXTUAL_MODEL = os.getenv("CONTEXTUAL_RETRIEVAL_MODEL", "gpt-4o-mini")

# 문서 성격이 다르므로 청킹 전략도 분리한다.
AX_CHUNK_SIZE = 900
AX_CHUNK_OVERLAP = 120
CURRICULUM_CHUNK_SIZE = 700
CURRICULUM_CHUNK_OVERLAP = 80

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "진단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


def _clean_text(text: str) -> str:
    # PDF/Excel에서 나온 들쭉날쭉한 공백을 정리해 검색 품질을 안정화한다.
    normalized_lines: list[str] = []
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


def _infer_curriculum_content_type(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("activity", "activities", "exercise", "workshop", "practice")):
        return "activity_plan"
    if any(keyword in lowered for keyword in ("goal", "goals", "objective", "learning outcome")):
        return "learning_goal"
    if any(keyword in lowered for keyword in ("case", "template", "agenda", "session")):
        return "session_plan"
    return "curriculum_reference"


def _format_excel_content(text: str) -> str:
    # Excel은 행/열 구조가 중요하므로 셀 경계를 읽기 쉽게 남긴다.
    rows: list[str] = []
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


def _tag_ax_type(metadata: dict[str, Any], text: str) -> None:
    # 검색 시 바로 필터링할 수 있도록 유형 정보를 메타데이터에도 넣는다.
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


def _create_source_doc_id(metadata: dict[str, Any]) -> str:
    base = "|".join(
        [
            str(metadata.get("source_file", "")),
            str(metadata.get("page_number", metadata.get("sheet_name", ""))),
            str(metadata.get("section_title", "")),
            str(metadata.get("doc_type", "")),
        ]
    )
    return _short_hash(base)


def _build_structured_document(content: str, metadata: dict[str, Any]) -> Document:
    # 청킹 전에 메타데이터를 본문 앞에 붙여 두면 잘린 청크도 문맥을 더 잘 유지한다.
    metadata = dict(metadata)
    metadata["source_doc_id"] = _create_source_doc_id(metadata)

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
    header_lines: list[str] = []
    for field_name in header_fields:
        value = metadata.get(field_name)
        if value not in (None, ""):
            header_lines.append(f"[{field_name}] {value}")
    header_lines.append("[content]")
    header_lines.append(content)
    return Document(page_content="\n".join(header_lines).strip(), metadata=metadata)


def _annotate_chunks(chunks: list[Document]) -> list[Document]:
    # base_content_hash는 원본 청크 기준으로 계산한다.
    # 이렇게 해 두면 contextual summary가 달라져도 증분 인덱싱 기준은 안정적으로 유지된다.
    source_counters: dict[tuple[str, str, str], int] = {}
    for chunk in chunks:
        source_key = (
            str(chunk.metadata.get("source_file", "")),
            str(chunk.metadata.get("page_number", chunk.metadata.get("sheet_name", ""))),
            str(chunk.metadata.get("section_title", "")),
        )
        chunk_index = source_counters.get(source_key, 0)
        source_counters[source_key] = chunk_index + 1

        base_content_hash = _short_hash(chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_id": f"{chunk.metadata.get('source_name', 'doc')}:{chunk_index}:{base_content_hash}",
                "base_content_hash": base_content_hash,
                "content_hash": base_content_hash,
                "word_count": len(chunk.page_content.split()),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "contextualized": False,
            }
        )
    return chunks


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
    return _annotate_chunks(splitter.split_documents(docs))


def _load_pdf_pages(file_path: str, metadata_builder) -> list[Document]:
    documents: list[Document] = []
    for page in PyPDFLoader(file_path).load():
        body = _clean_text(page.page_content)
        if not body:
            continue
        metadata = metadata_builder(page, body, file_path)
        documents.append(_build_structured_document(body, metadata))
    return documents


def load_ax_compass_documents() -> list[Document]:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    print("[RAG] Loading AXCompass PDF...")

    def _build_meta(page: Document, body: str, file_path: str) -> dict[str, Any]:
        source_file = page.metadata.get("source", file_path)
        page_number = _page_number_from_metadata(page.metadata.get("page"))
        section_title = _extract_section_title(body, f"ax_page_{page_number or 'unknown'}")
        metadata: dict[str, Any] = {
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
    documents: list[Document] = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".pdf") or filename == "AXCompass.pdf":
            continue

        file_path = os.path.join(DATA_DIR, filename)
        course_name = os.path.splitext(filename)[0]
        print(f"[RAG] Loading curriculum PDF: {filename}")

        def _build_meta(page: Document, body: str, pdf_path: str, course: str = course_name) -> dict[str, Any]:
            source_file = page.metadata.get("source", pdf_path)
            page_number = _page_number_from_metadata(page.metadata.get("page"))
            section_title = _extract_section_title(body, course)
            return {
                "doc_type": "curriculum_example",
                "source_file": source_file,
                "source_name": os.path.basename(source_file),
                "course_name": course,
                "page_number": page_number,
                "section_title": section_title,
                "module_name": section_title,
                "content_type": _infer_curriculum_content_type(body),
            }

        documents.extend(_load_pdf_pages(file_path, _build_meta))

    return documents


def load_curriculum_excel_documents() -> list[Document]:
    documents: list[Document] = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".xlsx"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        course_name = os.path.splitext(filename)[0]
        print(f"[RAG] Loading curriculum Excel: {filename}")

        for sheet_index, doc in enumerate(UnstructuredExcelLoader(file_path).load(), start=1):
            body = _format_excel_content(doc.page_content)
            if not body:
                continue

            source_file = doc.metadata.get("source", file_path)
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


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n..."


def _build_contextual_prompt(full_document: str, chunk_text: str, metadata: dict[str, Any]) -> str:
    # 질문을 직접 찾을 수 있도록 "문서 전체 대비 이 청크의 역할"만 짧게 생성한다.
    return f"""
You are preparing context for retrieval.
Write a concise Korean note, 2-3 sentences max.
Explain where this chunk fits in the full document and what information it is useful for.
Do not repeat the chunk verbatim.

[document type]
{metadata.get("doc_type", "")}

[section]
{metadata.get("section_title", "")}

[full document excerpt]
{_truncate_for_prompt(full_document, 2500)}

[chunk]
{_truncate_for_prompt(chunk_text, 1200)}
""".strip()


def _apply_contextual_retrieval(chunks: list[Document], source_lookup: dict[str, str]) -> list[Document]:
    if not USE_CONTEXTUAL_RETRIEVAL or not chunks:
        return chunks

    print(f"[RAG] Applying contextual retrieval to {len(chunks)} chunks...")
    llm = ChatOpenAI(
        model=CONTEXTUAL_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompts = [
        _build_contextual_prompt(
            source_lookup.get(chunk.metadata.get("source_doc_id", ""), chunk.page_content),
            chunk.page_content,
            chunk.metadata,
        )
        for chunk in chunks
    ]

    responses = llm.batch(prompts)
    for chunk, response in zip(chunks, responses):
        context_note = response.content.strip()
        if not context_note:
            continue

        original_text = chunk.page_content
        chunk.page_content = f"[context]\n{context_note}\n\n{original_text}"
        chunk.metadata.update(
            {
                "contextualized": True,
                "context_summary": context_note,
                "content_hash": _short_hash(chunk.page_content),
            }
        )

    print("[RAG] Contextual retrieval enrichment complete")
    return chunks


def _build_ax_compass_chunks() -> tuple[list[Document], dict[str, str]]:
    docs = load_ax_compass_documents()
    source_lookup = {doc.metadata["source_doc_id"]: doc.page_content for doc in docs}
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=AX_CHUNK_SIZE,
        chunk_overlap=AX_CHUNK_OVERLAP,
    )
    print(f"[RAG] AX Compass raw chunk count: {len(chunks)}")
    return chunks, source_lookup


def _build_curriculum_chunks() -> tuple[list[Document], dict[str, str]]:
    docs = load_curriculum_pdf_documents() + load_curriculum_excel_documents()
    source_lookup = {doc.metadata["source_doc_id"]: doc.page_content for doc in docs}
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=CURRICULUM_CHUNK_SIZE,
        chunk_overlap=CURRICULUM_CHUNK_OVERLAP,
    )
    print(f"[RAG] Curriculum example raw chunk count: {len(chunks)}")
    return chunks, source_lookup


def _collection_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


def _get_indexed_hashes(vectorstore: Chroma) -> set[str]:
    # 증분 인덱싱은 base_content_hash를 기준으로 본다.
    # contextual summary가 조금 달라도 원본 청크가 같으면 중복 추가하지 않는다.
    try:
        result = vectorstore._collection.get(include=["metadatas"])
    except Exception:
        return set()

    hashes: set[str] = set()
    for metadata in result.get("metadatas", []):
        if not metadata:
            continue
        stable_hash = metadata.get("base_content_hash") or metadata.get("content_hash")
        if stable_hash:
            hashes.add(stable_hash)
    return hashes


def _filter_new_documents(docs: list[Document], indexed_hashes: set[str]) -> list[Document]:
    return [
        doc
        for doc in docs
        if (doc.metadata.get("base_content_hash") or doc.metadata.get("content_hash", "")) not in indexed_hashes
    ]


def _ensure_index(vectorstore: Chroma, loader, label: str) -> None:
    # 첫 실행은 전체 인덱싱, 이후에는 새 청크만 추가하는 구조다.
    # 나중에 수정/삭제 동기화가 필요해져도 이 구조에서 확장하기 쉽다.
    existing_count = _collection_count(vectorstore)
    raw_chunks, source_lookup = loader()

    if existing_count == 0:
        chunks = _apply_contextual_retrieval(raw_chunks, source_lookup)
        print(f"[VectorDB] Indexing {label} collection...")
        if chunks:
            vectorstore.add_documents(chunks)
        print(f"[VectorDB] {label} indexed ({len(chunks)} chunks)")
        return

    indexed_hashes = _get_indexed_hashes(vectorstore)
    # 먼저 "새 청크인지" 판별하고, 그 뒤에만 context를 생성해야 불필요한 LLM 재호출이 없다.
    new_chunks = _filter_new_documents(raw_chunks, indexed_hashes)
    if new_chunks:
        new_chunks = _apply_contextual_retrieval(new_chunks, source_lookup)
        print(f"[VectorDB] Adding {len(new_chunks)} new {label} chunks...")
        vectorstore.add_documents(new_chunks)
        print(f"[VectorDB] {label} updated ({_collection_count(vectorstore)} chunks)")
    else:
        print(f"[VectorDB] No new {label} chunks. Keeping existing index ({existing_count} chunks)")


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

    _ensure_index(vectorstore, _build_ax_compass_chunks, "AX Compass")
    _ensure_index(vectorstore, _build_curriculum_chunks, "Curriculum examples")
    return vectorstore
