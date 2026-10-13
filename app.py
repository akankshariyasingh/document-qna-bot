
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader #for pdf loading
from langchain_text_splitters import RecursiveCharacterTextSplitter #for splitting text
from langchain_huggingface import HuggingFaceEmbeddings #for local embeddings
from langchain_google_genai import ChatGoogleGenerativeAI #for generating answers
from langchain_community.vectorstores import InMemoryVectorStore #for vector database


#chat model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

#embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#function to process the uploaded PDF
def document_process(path):

    #loading the document
    loader = PyPDFLoader(path)
    docs = loader.load()

    #splitting text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    #creating vector database
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vector_db


#function to answer user's question
def get_answer(query, vector_db):

    #searching for relevant text
    documents = vector_db.similarity_search(query, k=5)

    #joining the text
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    #sending question and document text to Gemini
    prompt = f"""
    Answer the question using the given document text.
    If the answer is not available, say so.
    Do not make up information.

    Document:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.text


#chatbot interface
st.title("Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

#checking whether a document is uploaded
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []


#first screen: upload PDF
if not st.session_state.uploaded:

    file = st.file_uploader("Choose your PDF", type="pdf")

    if file:
        with st.spinner("Reading your document..."):

            #saving PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temp_file:
                temp_file.write(file.getvalue())
                path = temp_file.name

            try:
                st.session_state.vector_db = document_process(path)
                st.session_state.uploaded = True
            finally:
                os.remove(path)

        st.rerun()


#second screen: ask questions
else:

    st.success("Your document is ready!")

    #showing previous questions and answers
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about your PDF")

    if query:

        #getting the answer
        with st.spinner("Finding the answer..."):
            answer = get_answer(
                query,
                st.session_state.vector_db
            )

        #saving the conversation
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.rerun()

    #upload a different PDF
    if st.button("Upload another document"):
        st.session_state.uploaded = False
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.rerun()
