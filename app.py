
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import time

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI


# API key configuration
try:
    if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass


# Load the Gemini model
@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_output_tokens=1000,
        timeout=60
    )


# Process the uploaded PDF
def document_process(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    document_text = "\n\n".join(
        doc.page_content for doc in docs
    ).strip()

    return document_text


# Format Gemini's response as readable text
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


# Answer the user's question
def get_answer(query, document_text):
    query = query.strip()

    if not query:
        return "Please enter a question about your document.", 0.0

    if not document_text.strip():
        return "No readable text was extracted from this PDF.", 0.0

    # Send only the first 10,000 characters to Gemini
    max_chars = 10000
    relevant_text = document_text[:max_chars]

    if len(document_text) > max_chars:
        relevant_text += (
            "\n\nNote: Only the beginning of the document was included. "
            "Information from later sections may be missing."
        )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using the supplied document text.

Instructions:
- Use clear, simple, natural language.
- Keep normal answers under 120 words.
- Answer directly without unnecessary introductions.
- Do not repeat the question.
- For a summary requested in 10 lines, provide 10 numbered lines.
- For key takeaways, use a numbered list.
- If the answer is not present in the supplied text, say so.
- Never invent facts.
- Return plain, readable text only.
- Do not return JSON, SVG, signatures, or metadata.

DOCUMENT:
{relevant_text}

USER QUESTION:
{query}

ANSWER:
"""

    # Retry temporary service-unavailable errors
    for attempt in range(3):
        try:
            start = time.perf_counter()
            response = llm.invoke(prompt)
            answer_time = time.perf_counter() - start

            answer = format_answer(response.content)

            if not answer:
                answer = (
                    "Gemini returned an empty text response. "
                    "Please try asking the question again."
                )

            return answer, answer_time

        except Exception as e:
            error_message = str(e).upper()

            if (
                ("503" in error_message or "UNAVAILABLE" in error_message)
                and attempt < 2
            ):
                time.sleep(2 ** (attempt + 1))
                continue

            raise


# Initialize Gemini
try:
    llm = load_model()
except Exception:
    st.error(
        "Could not initialize Gemini. Check your API key, "
        "model availability, and Streamlit Cloud Secrets."
    )
    st.stop()


# Chatbot interface
st.title("Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")


# Initialize session state
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "timings" not in st.session_state:
    st.session_state.timings = {}


# Screen 1: Upload a PDF
if not st.session_state.uploaded:
    file = st.file_uploader("Choose your PDF", type="pdf")

    if file is not None:
        with st.spinner("Reading your document..."):
            path = None

            try:
                # Save the PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(file.getvalue())
                    path = temp_file.name

                # Extract the PDF text once
                extracted_text = document_process(path)

                if not extracted_text:
                    st.error(
                        "No readable text was found in this PDF. "
                        "It may be scanned or image-based. "
                        "Please use a text-based PDF or add OCR support."
                    )
                else:
                    st.session_state.document_text = extracted_text
                    st.session_state.uploaded = True

            except Exception as e:
                st.error(f"Could not read the PDF: {e}")

            finally:
                if path and os.path.exists(path):
                    os.remove(path)

        if st.session_state.uploaded:
            st.rerun()


# Screen 2: Ask questions
else:
    st.success("Your document is ready!")

    # Display previous questions and answers
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

    if query and query.strip():
        query = query.strip()

        # Save the question
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
                error_message = str(e).upper()

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                ):
                    answer = (
                        "Your Gemini API quota has been reached. "
                        "Please wait for it to reset or check your usage at "
                        "https://ai.dev/rate-limit"
                    )

                elif (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):
                    answer = (
                        "Gemini is temporarily unavailable. "
                        "Please wait a little and try again."
                    )

                else:
                    answer = (
                        "Sorry, I could not generate an answer. "
                        f"Error: {e}"
                    )

        # Save the answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "question": query
            }
        )

        st.rerun()

    # Upload a different PDF
    if st.button("Upload another document"):
        st.session_state.uploaded = False
        st.session_state.messages = []
        st.session_state.document_text = ""
        st.session_state.timings = {}

        st.rerun()
