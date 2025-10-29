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
                 output_directory: str,
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

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cpu'}
        )
        self.db = Chroma.from_documents(chunks, embeddings)

    def search(self, query: str) -> list[str]:
        documents = self.db.similarity_search(query)
        return [d.page_content for d in documents]
