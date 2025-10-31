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
                 ):
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

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cpu'}
        )

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
