
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import time

import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# 1. API KEY CONFIGURATION


try:
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass



# 2. LOAD MODELS


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        max_output_tokens=500,
        timeout=60
    )

# 3. LOAD AND SPLIT PDF


def process_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    # Remove empty chunks
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    return chunks



# 4. CREATE FAISS VECTOR INDEX


def create_vector_index(chunks):
    embedding_model = load_embedding_model()

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# 5. RETRIEVE RELEVANT CHUNKS

def retrieve_chunks(question, chunks, index, top_k=4):
    embedding_model = load_embedding_model()

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for position, chunk_index in enumerate(indices[0]):
        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0) + 1,
            "score": float(scores[0][position])
        })

    return results


# 6. FORMAT GEMINI RESPONSE


def format_answer(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text_value = item.get("text")

                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())

        return "\n".join(parts).strip()

    if isinstance(content, dict):
        text_value = content.get("text")

        if isinstance(text_value, str):
            return text_value.strip()

    return ""


# 7. GENERATE ANSWER USING RETRIEVED CONTEXT

def get_answer(question, chunks, index):
    question = question.strip()

    if not question:
        return "Please enter a question.", 0.0, []

    start = time.perf_counter()

    # Retrieve relevant sections from the PDF
    results = retrieve_chunks(
        question,
        chunks,
        index,
        top_k=4
    )

    if not results:
        return "No relevant document sections were found.", 0.0, []

    context = "\n\n---\n\n".join(
        f"[PDF page {result['page']}]\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using only the retrieved document sections.

Instructions:
- Use simple, clear language.
- Keep normal answers concise, usually under 120 words.
- Answer directly without unnecessary introductions.
- Do not invent facts.
- If the supplied sections do not contain the answer,
  say that you could not find the answer in the retrieved sections.
- Cite supporting page numbers, such as (Page 54).
- Return plain, readable text only.

RETRIEVED DOCUMENT SECTIONS:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        llm = load_chat_model()
        response = llm.invoke(prompt)
        answer = format_answer(response.content)

        if not answer:
            answer = "Gemini returned no readable answer. Please try again."

        elapsed = time.perf_counter() - start

        source_pages = sorted({
            result["page"] for result in results
        })

        return answer, elapsed, source_pages

    except Exception:
        raise


# 8. STREAMLIT INTERFACE


st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄"
)

st.title("Document Q&A Chatbot")
st.write(
    "Upload a PDF and ask questions. "
    "RAG retrieves relevant sections to help answer your questions."
)


# Initialize session state
defaults = {
    "uploaded": False,
    "messages": [],
    "chunks": [],
    "vector_index": None,
    "document_name": "",
    "timings": {}
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# 9. UPLOAD AND PROCESS DOCUMENT


if not st.session_state.uploaded:
    uploaded_file = st.file_uploader(
        "Choose your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        path = None

        try:
            with st.spinner(
                "Extracting text, creating chunks and building the search index..."
            ):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    path = temp_file.name

                chunks = process_pdf(path)

                if not chunks:
                    st.error(
                        "No readable text was found in this PDF. "
                        "Please try a text-based PDF."
                    )
                else:
                    vector_index = create_vector_index(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.vector_index = vector_index
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.timings = {}
                    st.session_state.uploaded = True

        except Exception as e:
            st.error(f"Could not process the PDF: {e}")

        finally:
            if path and os.path.exists(path):
                os.remove(path)

        if st.session_state.uploaded:
            st.rerun()



# 10. CHAT WITH THE DOCUMENT

else:
    st.success(
        f"Document ready: {st.session_state.document_name}"
    )

    st.caption(
        f"Indexed {len(st.session_state.chunks)} text chunks for retrieval."
    )

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                question = message.get("question", "")
                timing = st.session_state.timings.get(question)

                if timing is not None:
                    st.caption(f"Response time: {timing:.2f} seconds")

                pages = message.get("sources", [])

                if pages:
                    page_text = ", ".join(
                        str(page) for page in pages
                    )
                    st.caption(f"Retrieved source pages: {page_text}")

    question = st.chat_input(
        "Ask a question about your PDF"
    )

    if question and question.strip():
        question = question.strip()

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Searching the document and generating an answer..."):
            try:
                answer, elapsed, sources = get_answer(
                    question,
                    st.session_state.chunks,
                    st.session_state.vector_index
                )

                st.session_state.timings[question] = elapsed

            except Exception as e:
                error_message = str(e).upper()

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                ):
                    answer = (
                        "Your Gemini API quota has been reached. "
                        "Please wait for it to reset or check "
                        "https://ai.dev/rate-limit"
                    )

                elif (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):
                    answer = (
                        "Gemini is temporarily unavailable. "
                        "Please try again later."
                    )

                else:
                    answer = (
                        "Sorry, I could not generate an answer. "
                        f"Error: {e}"
                    )

                elapsed = None
                sources = []

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": question,
                "sources": sources
            })

        st.rerun()

    # Start with another PDF
    if st.button("Upload another document"):
        st.session_state.uploaded = False
        st.session_state.messages = []
        st.session_state.chunks = []
        st.session_state.vector_index = None
        st.session_state.document_name = ""
        st.session_state.timings = {}

        st.rerun()
