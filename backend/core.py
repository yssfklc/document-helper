from typing import Any, Tuple, List
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from consts import INDEX_NAME
import streamlit as st
import os

pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    PINECONE_ENVIRONMENT_REGION=os.environ['PINECONE_ENVIRONMENT_REGION'],
)


def run_llm(query: str, chat_history: List[Tuple[str, Any]]) -> Any:
    embeddings = OpenAIEmbeddings(OPENAI_API_KEY=os.environ['OPENAI_API_KEY'])
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is langchain"))
