from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


st.header("Langchain Documentation Reader and Question Answerer")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_source_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    source_list = list(source_urls)
    source_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(source_list):
        sources_string += f"{i+1}. {source}"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['answer']} \n\n {create_source_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))
if st.session_state["chat_answer_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]
    ):
        message(user_query, is_user=True)
        message(generated_response)
