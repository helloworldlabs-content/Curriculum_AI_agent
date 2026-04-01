import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
PDF_PATH = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_v2"

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}


def _tag_ax_type(doc: Document) -> None:
    if doc.metadata.get("doc_type") != "ax_compass":
        return

    for type_name, info in TYPE_INFO.items():
        if type_name in doc.page_content:
            doc.metadata.update(
                {
                    "type_name": type_name,
                    "group": info["group"],
                    "english": info["english"],
                }
            )
            return


def load_ax_compass_documents() -> list[Document]:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일이 없습니다: {PDF_PATH}")

    print("[RAG] AXCompass PDF 로드 중...")
    documents = PyPDFLoader(PDF_PATH).load()
    for doc in documents:
        doc.metadata["doc_type"] = "ax_compass"
    return documents


def load_curriculum_pdf_documents() -> list[Document]:
    documents: list[Document] = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".pdf") or filename == "AXCompass.pdf":
            continue

        file_path = os.path.join(DATA_DIR, filename)
        course_name = os.path.splitext(filename)[0]
        print(f"[RAG] 커리큘럼 PDF 로드: {filename}")
        pages = PyPDFLoader(file_path).load()
        for page in pages:
            page.metadata.update(
                {
                    "doc_type": "curriculum_example",
                    "course_name": course_name,
                }
            )
        documents.extend(pages)

    return documents


def load_curriculum_excel_documents() -> list[Document]:
    documents: list[Document] = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".xlsx"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        course_name = os.path.splitext(filename)[0]
        print(f"[RAG] 커리큘럼 Excel 로드: {filename}")
        loaded_docs = UnstructuredExcelLoader(file_path).load()
        for doc in loaded_docs:
            doc.metadata.update(
                {
                    "doc_type": "curriculum_example",
                    "course_name": course_name,
                }
            )
        documents.extend(loaded_docs)

    return documents


def load_and_split_documents() -> list[Document]:
    all_docs = []
    all_docs.extend(load_ax_compass_documents())
    all_docs.extend(load_curriculum_pdf_documents())
    all_docs.extend(load_curriculum_excel_documents())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)

    for chunk in chunks:
        _tag_ax_type(chunk)

    ax_count = sum(1 for chunk in chunks if chunk.metadata.get("doc_type") == "ax_compass")
    example_count = len(chunks) - ax_count
    print(f"[RAG] 총 {len(chunks)}개 청크 (AX Compass: {ax_count}, 커리큘럼 예시: {example_count})")
    return chunks


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
        chunks = load_and_split_documents()
        vectorstore.add_documents(chunks)
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")

    return vectorstore
