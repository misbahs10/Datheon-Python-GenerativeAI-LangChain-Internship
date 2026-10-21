
import hashlib

import streamlit as st

from rag_pipeline import build_vector_store, answer_question


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Leaf | RAG Document Q&A",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# GLOBAL THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --sage: #52785e;
        --sage-dark: #234f40;
        --sage-soft: #e7eee3;
        --cream: #f8f7f2;
        --border: #dce5da;
        --muted: #63766d;
    }

    .stApp {
        background: var(--cream);
        color: #263e34;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 247, 242, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #edf1e8;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        color: var(--sage-dark) !important;
        letter-spacing: -0.5px;
    }

    p, label, li {
        color: #405b50;
    }

    .brand {
        color: #52785e;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2.2px;
        margin-bottom: 13px;
    }

    .hero-title {
        color: #234f40;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(34px, 4vw, 49px);
        font-weight: 700;
        letter-spacing: -1.7px;
        line-height: 1.08;
        margin: 8px 0 16px 0;
    }

    .hero-description {
        color: #63766d;
        font-size: 15px;
        line-height: 1.8;
        max-width: 640px;
    }

    .eyebrow {
        color: #52785e;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-top: 8px;
    }

    .hero-art {
        min-height: 205px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 4px auto;
    }

    .hero-art svg {
        max-width: 100%;
        height: auto;
    }

    .section-title {
        color: #234f40;
        font-size: 23px;
        font-weight: 750;
        letter-spacing: -0.6px;
    }

    .section-caption {
        color: #718077;
        font-size: 13px;
        line-height: 1.7;
        margin: 4px 0 17px 0;
    }

    .upload-panel {
        background: #ffffff;
        border: 1px solid #e6e9e1;
        border-radius: 23px;
        padding: 25px;
        box-shadow: 0 12px 34px rgba(47, 76, 55, 0.045);
    }

    .upload-heading {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 19px;
    }

    .upload-icon {
        display: flex;
        flex: 0 0 64px;
        width: 64px;
        height: 64px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #e6eee2;
        color: #285541;
    }

    .upload-heading-title {
        color: #234f40;
        font-size: 22px;
        font-weight: 750;
    }

    .upload-heading-subtitle {
        color: #718077;
        font-size: 14px;
        margin-top: 4px;
    }

    [data-testid="stFileUploader"] {
        background: #fbfcf9;
        border: 1px dashed #a9c1ad;
        border-radius: 17px;
        padding: 17px;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #52785e;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
        min-height: 160px;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #52785e;
        color: white;
        border: 1px solid #52785e;
        border-radius: 30px;
        padding: 7px 20px;
        font-weight: 650;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #234f40;
        color: white;
        border-color: #234f40;
    }

    .file-preview {
        display: flex;
        align-items: center;
        gap: 13px;
        background: #f6f8f3;
        border: 1px solid #d5e2d3;
        border-radius: 14px;
        padding: 15px;
        margin-top: 15px;
    }

    .pdf-icon {
        width: 43px;
        height: 48px;
        flex: 0 0 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        background: #fce9e5;
        color: #bd4b42;
        font-size: 13px;
        font-weight: 850;
        border: 1px solid #f2d4ce;
    }

    .file-name {
        color: #234f40;
        font-size: 13px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .file-size {
        color: #718077;
        font-size: 12px;
        margin-top: 4px;
    }

    .ready-pill {
        display: inline-block;
        background: #e1eddf;
        border: 1px solid #d0e1cd;
        border-radius: 24px;
        color: #316342;
        padding: 7px 11px;
        font-size: 11px;
        font-weight: 750;
        white-space: nowrap;
    }

    [data-testid="stMetric"] {
        background: #f6f8f3;
        border: 1px solid #edf0e8;
        border-radius: 15px;
        padding: 15px 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #718077 !important;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #234f40 !important;
    }

    .status-pill {
        display: inline-block;
        background: #e1eddf;
        border: 1px solid #d0e1cd;
        border-radius: 24px;
        padding: 6px 12px;
        color: #316342;
        font-size: 12px;
        font-weight: 750;
    }

    .sidebar-brand {
        background: white;
        border: 1px solid #dfe7dc;
        border-radius: 17px;
        padding: 17px;
        margin-bottom: 22px;
    }

    .sidebar-logo {
        color: #285541;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 29px;
        font-weight: 800;
    }

    .sidebar-caption {
        color: #63766d;
        font-size: 13px;
        margin-top: 3px;
    }

    .step-card {
        padding: 9px 0 12px 0;
        border-bottom: 1px solid #dce5da;
    }

    .step-title {
        color: #285541;
        font-size: 14px;
        font-weight: 750;
    }

    .step-description {
        color: #718077;
        font-size: 12px;
        line-height: 1.7;
        margin-top: 4px;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e9df;
        border-radius: 16px;
        padding: 14px 17px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background: #f0f5ed;
        border-color: #dce6d8;
    }

    [data-testid="stChatInput"] {
        border-color: #b8c9b4;
    }

    [data-testid="stChatInput"] textarea {
        background: white;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #dfe7dc;
        border-radius: 12px;
        margin: 7px 0;
    }

    .footer {
        color: #87938a;
        font-size: 11px;
        text-align: center;
        padding: 27px 0 12px 0;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid #cbd9c7;
        background: white;
        color: #345b43;
        font-weight: 650;
    }

    .stButton > button:hover {
        background: #e6eee2;
        border-color: #9db69c;
        color: #234f40;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 34px;
        }

        .upload-panel {
            padding: 15px;
            border-radius: 17px;
        }

        .upload-heading-title {
            font-size: 19px;
        }

        .hero-art {
            min-height: 100px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🌿 Leaf</div>
            <div class="sidebar-caption">RAG Document Q&A</div>
        </div>

        <div class="step-card">
            <div class="step-title">⌂ &nbsp; Workspace</div>
            <div class="step-description">Your document assistant</div>
        </div>

        <br>
        <div class="step-title">How it works</div>

        <div class="step-card">
            <div class="step-title">① &nbsp; Upload</div>
            <div class="step-description">
                Select a text-based PDF document.
            </div>
        </div>

        <div class="step-card">
            <div class="step-title">② &nbsp; Ask</div>
            <div class="step-description">
                Enter a question about its content.
            </div>
        </div>

        <div class="step-card">
            <div class="step-title">③ &nbsp; Verify</div>
            <div class="step-description">
                Review the answer and inspect retrieved source passages.
            </div>
        </div>

        <br>
        <div class="step-description">
            ♙ API credentials remain server-side.
            <br><br>
            ▤ Answers depend on your PDF.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
hero_left, hero_right = st.columns([1.45, 0.8], gap="medium")

with hero_left:
    st.markdown(
        """
        <div class="brand">🌿 LEAF · DOCUMENT INTELLIGENCE</div>
        <div class="hero-title">Your documents,<br>understood.</div>
        <div class="hero-description">
            Ask your documents anything. Upload a PDF, ask questions in
            natural language, and explore answers grounded in your document.
            Relevant source passages help you verify the information.
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        """
        <div class="hero-art">
          <svg viewBox="0 0 300 220" width="300"
               xmlns="http://www.w3.org/2000/svg" role="img"
               aria-label="PDF document and upload illustration">
            <circle cx="150" cy="108" r="94" fill="#e8eee3"/>
            <ellipse cx="150" cy="178" rx="90" ry="17" fill="#dce7d7"/>
            <path d="M202 49 C236 27 244 12 242 4 C216 13 200 29 202 49Z"
                  fill="#8ba98d"/>
            <path d="M214 73 C247 68 266 52 268 40 C243 42 222 55 214 73Z"
                  fill="#a8bda4"/>
            <path d="M207 98 C238 102 260 91 268 78 C243 72 220 82 207 98Z"
                  fill="#76977b"/>
            <path d="M211 72 C215 49 225 28 242 8"
                  fill="none" stroke="#52785e" stroke-width="2"/>
            <rect x="69" y="44" width="132" height="143"
                  rx="12" fill="#b9cdb6"/>
            <rect x="77" y="37" width="132" height="143"
                  rx="12" fill="#ffffff" stroke="#dce5da"/>
            <path d="M180 37 L209 66 L180 66Z" fill="#edf2e9"/>
            <rect x="94" y="66" width="76" height="8" rx="4" fill="#e5ebdf"/>
            <rect x="94" y="83" width="91" height="8" rx="4" fill="#e5ebdf"/>
            <rect x="94" y="100" width="64" height="8" rx="4" fill="#e5ebdf"/>
            <rect x="94" y="119" width="60" height="27" rx="5" fill="#52785e"/>
            <text x="124" y="137" fill="white" text-anchor="middle"
                  font-size="13" font-family="Arial" font-weight="700">PDF</text>
            <rect x="94" y="155" width="76" height="7" rx="3.5" fill="#e5ebdf"/>
            <circle cx="205" cy="153" r="30" fill="#ffffff"/>
            <circle cx="205" cy="153" r="23" fill="#52785e"/>
            <path d="M195 157 L195 153 A10 10 0 0 1 215 153
                     L215 157 M201 153 L205 148 L210 153
                     M205 148 L205 164"
                  fill="none" stroke="white" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# UPLOAD CARD
# --------------------------------------------------
st.markdown(
    '<div class="upload-panel">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="upload-heading">
        <div class="upload-icon">
            <svg width="34" height="34" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="1.8"
                 stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true">
                <path d="M12 16V4"/>
                <path d="M7 9l5-5 5 5"/>
                <path d="M20 16.5v2A2.5 2.5 0 0 1 17.5 21h-11A2.5 2.5 0 0 1 4 18.5v-2"/>
            </svg>
        </div>
        <div>
            <div class="upload-heading-title">Your document</div>
            <div class="upload-heading-subtitle">
                Start by uploading the PDF you want to explore.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Select a PDF containing readable text.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center;color:#718077;font-size:12px;
                    padding:3px 0 8px 0;">
            PDF files only &nbsp; · &nbsp; Upload a document to begin
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="footer">LEAF · Thoughtful document exploration</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------
pdf_bytes = uploaded_file.getvalue()
file_hash = "rag-v3-" + hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("document_hash") != file_hash:
    with st.spinner(
        "Preparing your document... The embedding model may download "
        "the first time."
    ):
        try:
            (
                client,
                collection,
                embedding_model,
                page_count,
                chunk_count,
            ) = build_vector_store(pdf_bytes)

            st.session_state.document_hash = file_hash
            st.session_state.collection = collection
            st.session_state.chroma_client = client
            st.session_state.embedding_model = embedding_model
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []

        except Exception as error:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(f"Could not process this PDF: {error}")
            st.stop()


# --------------------------------------------------
# FILE PREVIEW
# --------------------------------------------------
file_size = len(pdf_bytes)

if file_size < 1024 * 1024:
    size_label = f"{file_size / 1024:.1f} KB"
else:
    size_label = f"{file_size / (1024 * 1024):.2f} MB"

st.markdown(
    f"""
    <div class="file-preview">
        <div class="pdf-icon">PDF</div>
        <div style="flex:1;min-width:0;">
            <div class="file-name">{uploaded_file.name}</div>
            <div class="file-size">{size_label}</div>
        </div>
        <div class="ready-pill">✓ &nbsp; Document ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("▤  Text pages", st.session_state.page_count)

with m2:
    st.metric("▱  Text chunks", st.session_state.chunk_count)

with m3:
    st.metric(
        "☏  Conversation",
        len(st.session_state.get("messages", [])) // 2,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------
# CHAT
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Ask Leaf</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Ask a question about your document and review the passages '
    'that support the answer.'
    '</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.get("messages", []):
    avatar = "👤" if message["role"] == "user" else "🌿"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                st.markdown("**Sources used**")

                for index, source in enumerate(sources, start=1):
                    page = source.get("page", "?")
                    source_text = source.get("text", "")

                    with st.expander(f"📄 Source {index} · Page {page}"):
                        st.text(source_text)
                        st.caption("Retrieved passage from your PDF.")


# --------------------------------------------------
# QUESTION + ANSWER
# --------------------------------------------------
question = st.chat_input("Ask something about your PDF...")

if question and question.strip():
    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Searching your document..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state.collection,
                    st.session_state.embedding_model,
                )

                st.markdown(answer)

                if sources:
                    st.markdown("**Sources used**")

                    for index, source in enumerate(sources, start=1):
                        page = source.get("page", "?")
                        source_text = source.get("text", "")

                        with st.expander(
                            f"📄 Source {index} · Page {page}"
                        ):
                            st.text(source_text)
                            st.caption("Retrieved passage from your PDF.")
                else:
                    st.caption("No source passages were returned.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                error_message = (
                    f"Sorry, the question could not be processed: {error}"
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        🌿 LEAF · DOCUMENT INTELLIGENCE
        <br>
        Sage green · Thoughtful workflows · Document-grounded answers
    </div>
    """,
    unsafe_allow_html=True,
)
