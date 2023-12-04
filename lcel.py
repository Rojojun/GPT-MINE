from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(
    temperature=0.4
)

cache_dir = LocalFileStore("./.cache/")

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 비서입니다. 주어진 context만 사용하여 답변을 하시고, 모르면 모른다고 솔직하게 대답하세요. 없는 답변을 만들어 내지마세요. 답변의 끝마다 친절하게 이모지를 붙입니다.:\n\n{context}"),
    ("human", "{question}")
])

chain = {"context" : retriver, "question" : RunnablePassthrough(), "extra" : RunnablePassthrough() } | prompt | llm

chain.invoke("Victory Mansions에 대해 묘사하세요")