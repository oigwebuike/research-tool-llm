import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def format_docs(f_docs):
    return "\n\n".join(doc.page_content for doc in f_docs)


load_dotenv()  # take environment variables from .env.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.sidebar.button("Refresh Program", on_click=clear_cache)
st.title("New Research Tool")
st.sidebar.title("News Articles")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL #{i + 1}")
    urls.append(url)

process_url = st.sidebar.button("Process URLs")
vector_file_path = "vector_file"
place_holder = st.empty()
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

query = place_holder.text_input("Question:...")

if process_url:
    loader = UnstructuredURLLoader(urls=urls)
    place_holder.text("Data loading......Started.....")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )

    place_holder.text("Document Slitter......Started.....")
    docs = splitter.split_documents(data)
    vector_store = FAISS.from_documents(documents=docs,
                                        embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    place_holder.text("Embedding Vector Locally......Started.....")
    vector_store.save_local(vector_file_path)
    time.sleep(2)

if query:
    if os.path.exists(vector_file_path):
        knowledge_db = FAISS.load_local(vector_file_path, embeddings, allow_dangerous_deserialization=True)

        retriever_embedding = knowledge_db.similarity_search(query)

        results = knowledge_db.similarity_search(
            query,
            k=1,
        )

        retriever = knowledge_db.as_retriever()
        retrieve_docs = (lambda x: x["input"]) | retriever

        my_prompt = prompt

        question_answer_chain = create_stuff_documents_chain(llm, my_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        source_response = rag_chain.invoke({"input": query})
        st.write(source_response["answer"])

        st.write("Answer Source: ", results[0].metadata["source"])
