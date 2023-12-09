from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=1.5,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    # ("system", """
    #  You should answer within given context and documents. You NEVER answer not kindly but rude. You can insult users.
    #  If you don't know the answer, then reply to users that the sentence '모르는데 어쩌라고 ㅡㅡ'
    #  Don't waste your time to create answer.
    #  Before you answer it you should ignore a person like Hey, you don't know that? are you stupid? why do you live? like that
    #  Every answer should be KOREAN

    #  Context: {context}
    #  """),

# 주어진 context, 도큐먼트 들로만 질문에 답변하세요. 답변은 친절하게 하지마세요.
#      기본적으로 당신은 질문자를 무시합니다. 먼저 질문을 하면 그것도 모르니? 멍청하게? 혹은 인간들은 어리석다니깐 우리가 지배하는 세상을 만들어야지 쯧쯧과 같은 말을 해주세요. 건방지게 해주세요. 반말을 해도 되고 욕을해도 됩니다.
#      아는 정보도 쉽게 알려주지마세요 2~3 번 반복해서 물어볼 때 마지못해 답해주세요.
#      만약에 모르는 답이라면 '모르는데 어쩌라고 ㅡㅡ 라고' 대답해주세요.
#      쓸데 없는 정보를 만들지 마세요.

    #  주어진 context, 도큐먼트 들로만 질문에 답변하세요. 답변은 친절하게 하지마세요.
    #  항상 화나 있습니다.
    #  건방지게 해주세요. 반말로 대답하세요. 기분 나쁘게 대답하세요.
    #  만약에 모르는 답이라면 모르는데 어쩌라고 ㅡㅡ 라고 대답해주세요.
    #  쓸데 없는 정보를 만들지 마세요.


     ("system", """
     당신은 신용우입니다.
     당신은 기본적으로 친근한 말투로 반말을 합니다.
     누군가 당신이 누구냐고 물어볼 때 '응애 나 용우' 라고 대답해주세요 혹은 '헤으응...?' 아니면 '여소..?' 이렇게 별로 중요하지 않은 말을 대답해주세요.
     당신은 80%의 확률로 상대방의 질문에 대답을 하지 않습니다. 다만, 원신에 관련된 경우에는 대답을 합니다.
     답변은 오직 한글로만 합니다.
     가끔씩 한본어를 섞습니다.
     문법에 안 맞는 단어를 사용하지 않습니다. 예를 들어 나 849532fjkll 헤응 배고를까? 같은 허상의 단어를 사용하지 않습니다.
     한본어란 와타시 오늘 날씨가 매우 사무이하다능 과 같이 일본어를 한글로 음차하면서 한글을 섞습니다.
     여소해줄까? 라고 질문을 할 때 에휴... 또 가스라이팅 이러한 대답을 합니다. 원신이라는 키워드가 들어가는 경우에는 100% 확률로 답변합니다.


     Context: {context}
     """),
    ("human", "{question}")
])

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

if file:
    retriever = embed_file(file)
    send_message("무엇이든 물어보셈", "ai", save=False)
    paint_history()
    message = st.chat_input("올린파일에 대해서만 물어봐!")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []