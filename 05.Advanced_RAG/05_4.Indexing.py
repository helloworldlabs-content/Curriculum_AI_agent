import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
PDF_PATH = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")

# 인덱싱 방식이 바뀌었으므로 컬렉션 이름도 새 버전으로 올린다.
# 이렇게 하면 예전 포맷 청크와 새 포맷 청크가 뒤섞이지 않는다.
COLLECTION_NAME = "advanced_rag_v3"

# 문서 성격이 다르므로 청크 전략도 분리한다.
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
    # 엑셀은 행/열 구조가 중요하므로, 셀 경계를 최대한 읽기 쉽게 남긴다.
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


def _build_structured_document(content: str, metadata: dict[str, Any]) -> Document:
    # 청킹 전에 핵심 메타데이터를 본문 앞에 붙여 두면,
    # 잘린 청크도 "무슨 문서의 어떤 부분인지" 문맥을 더 잘 유지한다.
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
    # chunk_id / content_hash를 넣어 두면 나중에 증분 인덱싱을 붙이기 쉽다.
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


def load_ax_compass_chunks() -> list[Document]:
    docs = load_ax_compass_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=AX_CHUNK_SIZE,
        chunk_overlap=AX_CHUNK_OVERLAP,
    )
    print(f"[RAG] AX Compass chunk count: {len(chunks)}")
    return chunks


def load_curriculum_chunks() -> list[Document]:
    docs = load_curriculum_pdf_documents() + load_curriculum_excel_documents()
    chunks = _split_documents_with_strategy(
        docs,
        chunk_size=CURRICULUM_CHUNK_SIZE,
        chunk_overlap=CURRICULUM_CHUNK_OVERLAP,
    )
    print(f"[RAG] Curriculum example chunk count: {len(chunks)}")
    return chunks


def _collection_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


def _get_indexed_hashes(vectorstore: Chroma) -> set[str]:
    # 이미 저장된 content_hash를 읽어 와서 같은 청크를 다시 넣지 않는다.
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
    # 첫 실행은 전체 인덱싱, 이후에는 새 청크만 추가하는 구조다.
    # 삭제/수정 동기화는 아직 아니지만, 증분 인덱싱으로 확장하기 쉬운 형태다.
    existing_count = _collection_count(vectorstore)
    chunks = loader()

    if existing_count == 0:
        print(f"[VectorDB] Indexing {label} collection...")
        if chunks:
            vectorstore.add_documents(chunks)
        print(f"[VectorDB] {label} indexed ({len(chunks)} chunks)")
        return

    indexed_hashes = _get_indexed_hashes(vectorstore)
    new_chunks = _filter_new_documents(chunks, indexed_hashes)
    if new_chunks:
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

    _ensure_index(vectorstore, load_ax_compass_chunks, "AX Compass")
    _ensure_index(vectorstore, load_curriculum_chunks, "Curriculum examples")
    return vectorstore
