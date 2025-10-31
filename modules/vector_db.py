import os
from chromadb.api import CreateCollectionConfiguration
from chromadb.api.collection_configuration import \
    json_to_create_hnsw_configuration
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, \
    UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDb:
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self,
                 persist_directory: str | None = None,
                 input_directory: str | None = None,
                 input_document_path: str | None = None,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME,
                 chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP,
                 load_existing: bool = False,
                 ):
        """
        Initialize VectorDb.
        
        Args:
            persist_directory: Directory to persist the vector DB
            input_directory: Directory containing documents to load (if creating new DB)
            input_document_path: Single file path to load (if creating new DB)
            embedding_model_name: Name of the embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            load_existing: If True, load existing DB from persist_directory instead of creating new
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cpu'}
        )
        
        if load_existing:
            # Load existing vector DB from persistent directory
            if not os.path.exists(persist_directory):
                raise RuntimeError(f"Vector DB directory does not exist: {persist_directory}")
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            # Set chunk_number to None for existing DBs (not used in search anyway)
            self.chunk_number = None
            return  # Skip the rest of initialization if loading existing DB
        
        # Create new vector DB
        if input_directory:
            documents = DirectoryLoader(input_directory).load()
        elif input_document_path:
            documents = UnstructuredFileLoader(input_document_path).load()
        else:
            raise RuntimeError("You must specify either a directory or a file")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        ).split_documents(documents)

        self.chunk_number = len(chunks)
        print(f"Created {self.chunk_number} chunks")

        collection_config = CreateCollectionConfiguration(
            hnsw=json_to_create_hnsw_configuration(
                {
                    "space": "cosine",
                    "ef_construction": 250,
                    "ef_search": self.chunk_number,
                }
            )
        )
        self.db = Chroma.from_documents(chunks,
                                        embeddings,
                                        collection_configuration=collection_config,
                                        persist_directory=persist_directory,
                                        )

    def search(self, query: str) -> list[tuple[str, float]]:
        results = self.db.similarity_search_with_score(query, k=10)
        return [(doc.page_content, score) for doc, score in results]

    def close(self):
        """
        Explicitly close ChromaDB connections and release file handles.
        This is important on Windows where file handles must be explicitly released.
        Only call this when you want to delete the database directory.
        """
        if hasattr(self, 'db') and self.db is not None:
            try:
                # Clear the database reference to help GC release file handles
                # ChromaDB doesn't always expose a close method, but clearing
                # references helps garbage collection release file handles on Windows
                self.db = None
            except Exception:
                # If closing fails, don't raise - just try to clear references
                pass
