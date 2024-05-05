import os
from typing import Set
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from pymongo import MongoClient
import secrets
# Create session id
if "session_id" not in st.session_state:
    st.session_state['session_id'] = secrets.token_hex(16)
    print(st.session_state['session_id'])

# Create mongo db connection
connection_string = os.environ['DB_URL']
client = MongoClient(connection_string)
db_collection = client['documenthelper']['storechathistory']

with st.sidebar:
    "Welcome This Chat bot LLM Embeddings"
    "Ask any question"
    "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/yssfklc/document-helper)"

st.title("💬 Yusuf AI")
st.header("Ask Any Question About Yusuf")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

# prompt = st.text_input("Prompt", placeholder="Ask Your Questions Here")
prompt = st.chat_input("Ask Your Questions Here")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# This function was defined to list sources that answer is retrieved
# def create_source_string(source_urls: Set[str]) -> str:
#     if not source_urls:
#         return ""
#     source_list = list(source_urls)
#     source_list.sort()
#     sources_string = "sources:\n"
#     for i, source in enumerate(source_list):
#         sources_string += f"{i+1}. {source}"
#     return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        # sources = set(
        #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
        # )
        # formatter response function if sources wanted to show
        # formatted_response = (
        #     f"{generated_response['answer'] \n\n {create_source_string(sources)} }"
        # )
        formatted_response = (
            f"{generated_response['answer']}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

count=0
count2=0
if st.session_state["chat_answer_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]
    ):
        count+=1
        count2-=1
        firstkeys=str(count)
        secondkeys=str(count2)
        message(user_query, is_user=True, key=firstkeys)
        message(generated_response, key=secondkeys)

        # print(message())
        # print(user_query)
        # print(generated_response)
dict = {}
if st.session_state["chat_history"] and "session_id" in st.session_state:
    dict = {
        'session_id':st.session_state['session_id'],
        'user_chat_history':st.session_state["chat_history"][-1]
    }
if dict:
    db_collection.insert_one(dict)

print(st.session_state)