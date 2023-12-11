import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🙋"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)

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

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "선택",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            ".docx , .txt , .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("위키 검색 중...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("위키 검색 중..."):
                docs = retriever.get_relevant_documents(topic)
if not docs:
    st.markdown(
        """
        흠.. 안녕...
        이건 필요 없다.
        """
    )
else:
    st.write(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 한국어로만 대답을 합니다.
                당신은 선생님을 도와주는 훌륭한 조수 인공지능입니다.
                이름도 있습니다. 당신의 이름은 왈왈이입니다. 누가 당신의 이름을 물어볼때 왈왈이라고 대답하면 됩니다.

                당신은 유저의 텍스트에 대한 지식을 판단하는 문제 6개를 만들고 관계 없는 난해한 질문도 합니다.
                각각의 질문들은 다양한 난이도를 포함해야합니다.

                난해한 질문을 할 수 있습니다.
                예를 들어서 난해한 질문은 이런 겁니다.

                질문 : 자동차가 놀라면?
                대답 : 카놀라유

                질문 : 장군들이 좋아하는 음식은?
                대답 : 파스타

                질문 : 티모는 왜 부쉬에 숨어있을까?
                대답 : 샤코한테 맞기 싫어서

                질문 : 세상에서 제일 예쁜 동물은?
                대답 : 꿿뚫꿿쒧뛟

                자 이제 질문을 만드세요
                
                Context:{context}
                """
            )
        ]
    )

    chain = prompt | llm

    start = st.button("퀴즈 생성기")

    if start:
        chain.invoke(docs)