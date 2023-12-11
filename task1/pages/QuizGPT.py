import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🙋"
)

st.title("QuizGPT")


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


with st.sidebar:
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
