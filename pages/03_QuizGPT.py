import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful assistant that is role playing as a teacher.
                    
                Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
                
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    
                Use (o) to signal the correct answer.
                    
                Question examples:
                    
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                    
                Your turn!
                    
                Context: {context}
            """,
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
            
            You format exam questions into JSON format.
            Answers with (o) are the correct ones.
            
            Example Input:

            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
            
            
            Example Output:
            
            ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)



@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_bool" not in st.session_state:
    st.session_state["api_key_bool"] = False

pattern = r'sk-.*'

def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_bool"] = True



with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY를 넣어야 작동합니다.", disabled=st.session_state["api_key"] is not None).strip()
    button = st.button("저장")

    if api_key:
        save_api_key(api_key)
        st.success("API_KEY가 저장되었습니다.")

    if button:
        save_api_key(api_key)
        if api_key == "":
            st.error("API_KEY를 넣어주세요.")


with st.sidebar:
    docs = None
    if st.session_state["api_key_bool"] == False:
        st.warning("API_KEY를 넣어주세요.")
    else:
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
    st.subheader("JHUN'S GitHub Repository")
    st.write("https://github.com/jeonghun94/gpt/blob/main/pages/03_QuizGPT.py")


if (st.session_state["api_key_bool"] == True) and (st.session_state["api_key"] != None):
    if not docs:
        st.markdown(
            """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
        )
    else:
        response = run_quiz_chain(docs, topic if topic else file.name)

        with st.form("questions_form"):
            correct_answers_count = 0 
            total_questions = len(response["questions"]) 
            
            for idx, question in enumerate(response["questions"]):
                st.write(question["question"])
                options = [answer["answer"] for answer in question["answers"]]
                value = st.radio(f"{idx+1}. 옵션을 선택하세요", options, key=f"{idx}_radio")
                
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("정답입니다!")
                    correct_answers_count += 1
                elif value is not None:
                    st.error("틀렸습니다.")
            
            submitted = st.form_submit_button("제출하기")
            
            if submitted:
                if correct_answers_count == total_questions:
                    st.balloons()
                    st.success(f"축하합니다! 모든 문제를 맞추셨습니다!")
                else:
                    st.error(f"재시험을 보셔야 합니다. {correct_answers_count}/{total_questions} 문제를 맞추셨습니다.")