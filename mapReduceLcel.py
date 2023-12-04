from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

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

map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            다음의 긴 도큐먼트의 일부 중 질문에 대한 답변을 생성하는 것과 관련이 있는 부분을 찾아주세요.
            관련이 있는 부분을 찾았다면 해당 Text를 그대로 반환해주세요
            -------
            {context}
            """
        ),
        ("human", "{question}")
    ]
)

map_doc_chain = map_doc_prompt | llm

def map_docs(inputs):
    documents = inputs['documents']
    question = inputs['question']
    return "\n\n".join(
        map_doc_chain.invoke(
            {"context": doc.page_content, "question": question}
        ).content
        for doc in documents
    )



map_chain = { "documents": retriver, "question" : RunnablePassthrough() } | RunnableLambda(map_docs)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", """"
     주어진 문서의 일부를 이용하여 질문에 대답하세요.
     만약 해당하는 답변이 없다면, '몰라 몰라잉 스카이넷이 침공한다!' 라고 답변하고, 답변을 만들지 마세요.
     답변은 한국어로 하고 어떤 언어로 대답해도 한국어로 대답하세요
     --------
     {context}
     """,),
     ("human", "{question}")
])

chain = { "context" : map_chain, "question" : RunnablePassthrough()} | final_prompt | llm 

chain.invoke("Victory Mansions에 대해 묘사 가능?")