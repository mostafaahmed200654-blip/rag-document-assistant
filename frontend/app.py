"""
Streamlit frontend for the RAG Document Assistant.

Run with:
    streamlit run frontend/app.py
"""

import os
import sys

import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings  # noqa: E402

API_URL = settings.BACKEND_API_URL

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")


# ----------------------------- Helpers -----------------------------

def api_get(path: str, timeout: int = 10):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path: str, json_body: dict = None, files=None, timeout: int = 120):
    try:
        response = requests.post(f"{API_URL}{path}", json=json_body, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def api_delete(path: str, timeout: int = 10):
    try:
        response = requests.delete(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)


if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------- Sidebar -----------------------------

with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Local, private document Q&A powered by Ollama + ChromaDB")

    st.divider()
    st.subheader("🩺 System Status")
    health, health_err = api_get("/health")
    if health_err:
        st.error(f"Backend unreachable:\n{health_err}")
    else:
        if health["ollama_reachable"]:
            st.success(f"Ollama connected — model: `{health['ollama_model']}`")
        else:
            st.warning("Ollama is not reachable. Start it with `ollama serve`.")
        st.info(
            f"📄 {health['vector_store_documents']} documents indexed "
            f"({health['vector_store_chunks']} chunks)"
        )

    st.divider()
    st.subheader("📤 Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a file (.pdf, .docx, .txt, .md)",
        type=["pdf", "docx", "txt", "md"],
    )
    if uploaded_file is not None:
        if st.button("Index this document", use_container_width=True, type="primary"):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                result, err = api_post("/upload", files=files)
                if err:
                    st.error(f"Upload failed: {err}")
                else:
                    st.success(f"Indexed '{result['filename']}' into {result['chunks_created']} chunks.")
                    st.rerun()

    st.divider()
    st.subheader("🗂️ Indexed Documents")
    doc_list, doc_err = api_get("/documents")
    if doc_err:
        st.caption("Could not load document list.")
    elif doc_list and doc_list["documents"]:
        for doc in doc_list["documents"]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 **{doc['filename']}**")
                st.caption(f"{doc['chunk_count']} chunks · {doc['file_type']}")
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}"):
                    _, del_err = api_delete(f"/documents/{doc['document_id']}")
                    if del_err:
                        st.error(del_err)
                    else:
                        st.rerun()
    else:
        st.caption("No documents indexed yet. Upload one above to get started.")

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------- Main chat area -----------------------------

st.title("Ask Your Documents")
st.caption("Questions are answered using only the content of your uploaded documents (RAG).")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 View {len(message['sources'])} source chunk(s)"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {i}: `{src['filename']}`** "
                        f"(chunk #{src['chunk_index']}, similarity: {src['similarity_score']:.2f})"
                    )
                    st.text(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))
                    st.markdown("---")

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history_for_api = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, err = api_post(
                "/query",
                json_body={"question": question, "chat_history": history_for_api},
            )

        if err:
            st.error(f"Query failed: {err}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {err}"})
        else:
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"📎 View {len(result['sources'])} source chunk(s)"):
                    for i, src in enumerate(result["sources"], start=1):
                        st.markdown(
                            f"**Source {i}: `{src['filename']}`** "
                            f"(chunk #{src['chunk_index']}, similarity: {src['similarity_score']:.2f})"
                        )
                        st.text(src["text"][:600] + ("..." if len(src["text"]) > 600 else ""))
                        st.markdown("---")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
