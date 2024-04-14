from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
import streamlit as st
import openai as client
import re


print(client.__version__)

assistant_pattern = r'asst_.*'
api_pattern = r'sk-.*'

st.set_page_config(page_title="Assistant GPT", page_icon="🖥️")

api_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for a given query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_duck_duck_go",
            "description": "Search DuckDuckGo for a given query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
]

def search_wikipedia(query):
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

def search_duck_duck_go(query):
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)

def append_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def display_message(message, role, persist=True):
    with st.chat_message(role):
        st.markdown(message)
    if persist:
        append_message(message, role)

def create_assistant(api_key):
    return client.beta.assistants.create(
        name="Search Assistant",
        instructions="Search and return results for user queries",
        model="gpt-4-1106-preview",
        tools=api_tools,
    )        

def append_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def display_message(message, role, persist=True):
    with st.chat_message(role):
        st.markdown(message)
    if persist:
        append_message(message, role)

def replay_conversation():
    for message in st.session_state.get("messages", []):
        display_message(message["message"], message["role"], persist=False)

with st.sidebar:
    api_key = st.text_input("OpenAI API 키를 입력하세요")
    api_key_button = st.button("API 키 저장")
    if api_key_button:
        if api_key and re.match(api_pattern, api_key):
            create_assistant(api_key)
            st.session_state["api_key"] = api_key
            st.success("API 키가 성공적으로 저장되었습니다.")
        else:
            st.error("잘못된 API 키입니다.")

    assistant_id = st.text_input("어시스턴트 ID를 입력하세요")
    assistant_id_button = st.button("어시스턴트 ID 저장")
    if assistant_id_button:
        if assistant_id and re.match(assistant_pattern, assistant_id):
            st.session_state["assistant_id"] = assistant_id
            st.success("어시스턴트 ID가 성공적으로 저장되었습니다.")
        else:
            st.error("잘못된 어시스턴트 ID입니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state.get("assistant_id"):
    display_message("준비가 되었습니다! 질문해주세요.", "assistant", persist=False)
    replay_conversation()
    user_message = st.text_input("질문을 입력하세요", key="query", value="")
    if st.button("질문하기"):
        if user_message:
            thread = client.beta.threads.create(messages=[{"role": "user", "content": user_message}])
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=st.session_state["assistant_id"])
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for message in reversed(list(messages)):
                display_message(message.content[0].text.value, message.role)
else:
    st.warning("어시스턴트 ID가 없습니다. 입력하고 저장해주세요.")


with st.sidebar:
    st.subheader("JHUN'S GitHub Repository")
    st.write("https://github.com/jeonghun94/gpt/blob/main/pages/07_SiteGPT.py")    