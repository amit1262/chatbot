from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_vector_store(index_params: dict):
    """Generates a vector index from the provided documents.

    Args:
        index_params (dict): dict of parameters for indexing

    Returns:
        FAISS: index of the documents.
    """
    INDEX_OUTPUT_PATH = index_params.get("output_path")
    # embedding model
    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
    model = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    print(f"Successfully initialized embedding model: {EMBEDDING_MODEL_NAME}")

    # Check if the vector index already exists
    if os.path.exists(INDEX_OUTPUT_PATH):
        print("Loading existing vector index...")
        vector_store = FAISS.load_local(
            INDEX_OUTPUT_PATH, embeddings=model, allow_dangerous_deserialization=True
        )
    else:
        print(f"Creating a new vector index at: {INDEX_OUTPUT_PATH}\n")
        # get chunks of text from the documents
        CHUNK_SIZE = index_params.get("chunk_size", 1000)
        CHUNK_OVERLAP = index_params.get("chunk_overlap", 200)
        SRC_DOC_PATHS = index_params.get("doc_paths")
        # Load the PDF documents
        docs = []
        for path, page_range in SRC_DOC_PATHS:
            print(f"Loading document: {path}")
            try:
                loader = PyMuPDFLoader(path)
                if page_range:
                    partial_docs = loader.load()
                    start, end = page_range
                    filtered = [
                        doc
                        for doc in partial_docs
                        if start <= doc.metadata.get("page", -1) <= end
                    ]
                    docs.extend(filtered)
                else:
                    docs.extend(loader.load())
            except Exception as e:
                raise RuntimeError(f"Error loading document {path}: {e}")

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Total chunks created: {len(chunks)}")

        vector_store = FAISS.from_documents(chunks, model)
        vector_store.save_local(INDEX_OUTPUT_PATH)
        print(f"Vector index created and saved successfully at: {INDEX_OUTPUT_PATH}.")

    return vector_store


index_params = {
    "output_path": "../data/vector_index/ind",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "doc_paths": [
        ("../docs/OHDSI/TheBookOfOhdsi.pdf", (100, 106)),  # document, page range
        ("../docs/OHDSI/TheBookOfOhdsi.pdf", (173, 280)),
        ("../docs/Strategus/Strategus.pdf", None),
        ("../docs/Strategus/ExecuteStrategus.pdf", None),
        ("../docs/Strategus/IntroductionToStrategus.pdf", None),
        ("../docs/Strategus/CreatingAnalysisSpecification.pdf", None),
    ],
}

vector_store = get_vector_store(index_params=index_params)
