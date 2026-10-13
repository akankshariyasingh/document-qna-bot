
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile
import time

from langchain_community.document_loaders import PyPDFLoader #for pdf loading
from langchain_google_genai import ChatGoogleGenerativeAI #for generating answers



#chat model
@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        max_output_tokens=300,
        timeout=60
    )

llm = load_model()


#function to process the uploaded PDF
def document_process(path):

    #loading the document
    loader = PyPDFLoader(path)
    docs = loader.load()

    #extracting text from the document
    document_text = "\n\n".join(
        doc.page_content for doc in docs
    )

    return document_text


#function to answer user's question
def get_answer(query, document_text):

    prompt = f"""
Answer the question using only the given document text.
If the answer is not available, say so.
Do not make up information.
Keep the answer clear and concise.

Document:
{document_text}

Question: {query}

Answer:
"""

    # Generate the answer using Gemini
    start = time.perf_counter()
    response = llm.invoke(prompt)
    answer_time = time.perf_counter() - start

    return response.content, answer_time


#chatbot interface
st.title("Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")


#checking whether a document is uploaded
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "timings" not in st.session_state:
    st.session_state.timings = {}


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
                st.session_state.document_text = document_process(path)

                if not st.session_state.document_text.strip():
                    st.error(
                        "No readable text was found in this PDF. "
                        "Please upload a text-based PDF."
                    )
                else:
                    st.session_state.uploaded = True

            except Exception as e:
                st.error(f"Could not read the PDF: {e}")

            finally:
                if os.path.exists(path):
                    os.remove(path)

        if st.session_state.uploaded:
            st.rerun()


#second screen: ask questions
else:

    st.success("Your document is ready!")

    #showing previous questions and answers
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                timing = st.session_state.timings.get(
                    message.get("question", "")
                )

                if timing is not None:
                    st.caption(
                        f"Gemini response: {timing:.2f} seconds"
                    )

    query = st.chat_input("Ask a question about your PDF")

    if query:

        #getting the answer
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.spinner("Finding the answer..."):
            try:
                answer, answer_time = get_answer(
                    query,
                    st.session_state.document_text
                )

                st.session_state.timings[query] = answer_time

            except Exception as e:
                answer = (
                    "Sorry, I could not generate an answer. "
                    f"Please try again. Error: {e}"
                )

        #saving the conversation
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "question": query
            }
        )

        st.rerun()


    #upload a different PDF
    if st.button("Upload another document"):
        st.session_state.uploaded = False
        st.session_state.messages = []
        st.session_state.document_text = ""
        st.session_state.timings = {}
        st.rerun()
