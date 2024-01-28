import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

pc = pinecone.Pinecone(
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY"),
    PINECONE_ENVIRONMENT_REGION=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is langchain"))
