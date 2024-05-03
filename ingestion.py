import os
from langchain_community.document_loaders import ReadTheDocsLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from consts import INDEX_NAME
import streamlit as st
import os

pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    PINECONE_ENVIRONMENT_REGION=os.environ['PINECONE_ENVIRONMENT_REGION'],
)


def ingest_docs() -> None:
    # loader = ReadTheDocsLoader(path="langchain-docs/")
    loader = TextLoader(file_path='langchain-docs/biography.txt')
    raw_documents = loader.load()
    # raw_documents = loader.load()
    print(raw_documents)
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
