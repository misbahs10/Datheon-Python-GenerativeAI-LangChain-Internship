
import hashlib
from collections import defaultdict

import streamlit as st

from rag_pipeline import build_vector_store, answer_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A")
st.write(
    "Apni PDF upload karein aur us document ke content se related sawalat poochein."
)


# --------------------------------------------------
# PDF upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file is None:
    st.info("Shuru karne ke liye PDF upload karein.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
file_hash = "rag-v3-" + hashlib.sha256(pdf_bytes).hexdigest()


# --------------------------------------------------
# Process PDF only when the file changes
# --------------------------------------------------

if st.session_state.get("document_hash") != file_hash:
    with st.spinner(
        "PDF process ho rahi hai. Pehli dafa embedding model download ho sakta hai..."
    ):
        try:
            (
                client,
                collection,
                embedding_model,
                page_count,
                chunk_count
            ) = build_vector_store(pdf_bytes)

            st.session_state["document_hash"] = file_hash
            st.session_state["collection"] = collection
            st.session_state["chroma_client"] = client
            st.session_state["embedding_model"] = embedding_model
            st.session_state["page_count"] = page_count
            st.session_state["chunk_count"] = chunk_count

            # New document means new chat history.
            st.session_state["messages"] = []

        except Exception as error:
            st.error(f"PDF process nahi ho saki: {error}")
            st.stop()


# --------------------------------------------------
# Document status
# --------------------------------------------------

st.success(
    f"PDF ready! Pages with text: {st.session_state['page_count']} | "
    f"Text chunks: {st.session_state['chunk_count']}"
)

st.caption(
    "Scanned ya image-only PDFs ke liye OCR ki zaroorat ho sakti hai."
)


# --------------------------------------------------
# Clear chat button
# --------------------------------------------------

if st.button("🗑️ Clear chat history"):
    st.session_state["messages"] = []
    st.rerun()


# --------------------------------------------------
# Display existing conversation
# --------------------------------------------------

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                # Combine source chunks from the same page.
                sources_by_page = defaultdict(list)

                for source in sources:
                    page = source.get("page", "?")
                    text = source.get("text", "").strip()

                    if text:
                        sources_by_page[page].append(text)

                if sources_by_page:
                    with st.expander("📚 View sources"):
                        for page, texts in sorted(
                            sources_by_page.items(),
                            key=lambda item: str(item[0])
                        ):
                            st.markdown(f"**Source: Page {page}**")

                            # Avoid duplicate source text.
                            unique_texts = list(dict.fromkeys(texts))

                            for source_text in unique_texts:
                                st.write(source_text)


# --------------------------------------------------
# Ask a question
# --------------------------------------------------

question = st.chat_input(
    "Document ke baare mein sawal poochein..."
)

if question and question.strip():
    question = question.strip()

    # Store the user message once.
    st.session_state["messages"].append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Relevant information search ho rahi hai..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state["collection"],
                    st.session_state["embedding_model"]
                )

                if not answer:
                    answer = (
                        "Is document se koi answer generate nahi ho saka."
                    )

                st.markdown(answer)

                if sources:
                    sources_by_page = defaultdict(list)

                    for source in sources:
                        page = source.get("page", "?")
                        text = source.get("text", "").strip()

                        if text:
                            sources_by_page[page].append(text)

                    if sources_by_page:
                        with st.expander("📚 View sources"):
                            for page, texts in sorted(
                                sources_by_page.items(),
                                key=lambda item: str(item[0])
                            ):
                                st.markdown(f"**Source: Page {page}**")

                                unique_texts = list(dict.fromkeys(texts))

                                for source_text in unique_texts:
                                    st.write(source_text)
                else:
                    st.caption("Is jawab ke liye koi source passage return nahi hua.")

                # Store the assistant response once.
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Question process nahi ho saka: {error}")
