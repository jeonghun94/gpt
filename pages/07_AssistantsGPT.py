from langchain.utilities import DuckDuckGoSearchAPIWrapper
from openai import OpenAI
import streamlit as st
import json
import time

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def get_run(client, run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)

def create_message(client, thread_id, content, file_ids=None):
    return client.beta.threads.messages.create(thread_id=thread_id, role="user", content=content, file_ids=file_ids)

def get_messages(client, thread_id):
    messages = list(client.beta.threads.messages.list(thread_id=thread_id))
    messages.reverse()
    for message in messages:
        if message.role == "user":
            send_message(message.content[0].text.value, "user")

def get_tool_outputs(client, run_id, thread_id):
    run = get_run(client, run_id, thread_id)
    outputs = [
        {"output": functions_map[action.function.name](json.loads(action.function.arguments)),
        "tool_call_id": action.id}
        for action in run.required_action.submit_tool_outputs.tool_calls
    ]
    return outputs

def submit_tool_outputs(client, run_id, thread_id):
    print(client)
    print(run_id)
    print(thread_id)
    outputs = get_tool_outputs(client, run_id, thread_id)
    send_message("정보를 찾았어요", "ai")
    send_message(outputs[0]["output"], "ai")
    return client.beta.threads.runs.submit_tool_outputs(run_id=run_id, thread_id=thread_id, tool_outputs=outputs)

def wait_on_run(client, run, thread):
    while run.status in ["queued", "in_progress"]:
        run = get_run(client, run.id, thread.id)
        time.sleep(0.5)
    return run

ddg = DuckDuckGoSearchAPIWrapper()

def get_response(category_data):
    return ddg.run(category_data.get("category", ""))

functions_map = {
    "get_response": get_response,
}

st.set_page_config(page_title="AssistantsGPT", page_icon="🤖")
st.markdown("# 🤖 AssistantsGPT - Final")

api_key = st.sidebar.text_input("API Key를 입력하세요.")
if api_key and api_key.startswith("sk-"):
    st.session_state["api_key"] = api_key
    client = OpenAI(api_key=api_key)
    assistant_id = "asst_zGNxVcPHD8OLwp6hhdt5WenV"
    category = st.text_input("대화를 시작하세요.")
    if category:
        thread = client.beta.threads.create(messages=[{"role": "user", "content": f"I want to know {category}"}])
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        run = wait_on_run(client, run, thread)
        if run:
            send_message("잠시만 기다려 주세요.", "ai", save=False)
            paint_history()
            get_messages(client, thread.id)
            submit_tool_outputs(client, run.id, thread.id)

with st.sidebar:
    st.subheader("JHUN'S GitHub Repository")
    st.write("https://github.com/jeonghun94/gpt/blob/main/pages/07_AssistantsGPT.py")