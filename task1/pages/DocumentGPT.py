from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

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
    s = retriever.invoke("winston")
    s