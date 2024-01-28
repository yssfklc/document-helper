import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from consts import INDEX_NAME

pc = pinecone.Pinecone(
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY"),
    PINECONE_ENVIRONMENT_REGION=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) } documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"loaded {len(raw_documents)} documents")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("Added to Pinecone vectorstore")


if __name__ == "__main__":
    ingest_docs()
