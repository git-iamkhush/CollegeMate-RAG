import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CollegeMate",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 CollegeMate")
st.subheader("AI-Powered College Information Assistant")
st.write(
    "Ask questions about academic rules, courses, placements, "
    "library services and student support."
)


# =========================================================
# GEMINI API
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key is not configured.")
    st.stop()

client = genai.Client(api_key=api_key)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

@st.cache_data
def load_documents():

    documents = []

    data_folder = Path("data")

    # Load TXT files
    for file in data_folder.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as f:

            documents.append({
                "text": f.read(),
                "source": file.name,
                "page": None
            })

    # Load PDFs if the PDFs folder exists
    pdf_folder = data_folder / "PDFs"

    if pdf_folder.exists():

        for file in pdf_folder.glob("*.pdf"):

            reader = PdfReader(str(file))

            for page_number, page in enumerate(
                reader.pages, start=1
            ):

                page_text = page.extract_text()

                if page_text:

                    documents.append({
                        "text": page_text,
                        "source": file.name,
                        "page": page_number
                    })

    return documents


documents = load_documents()


# =========================================================
# CHUNKING
# =========================================================

@st.cache_data
def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    all_chunks = []

    for doc in documents:

        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

    return all_chunks


all_chunks = create_chunks(documents)

chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# CREATE FAISS INDEX
# =========================================================

@st.cache_resource
def create_faiss_index(chunk_texts):

    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


index = create_faiss_index(chunk_texts)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_documents(query, k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    results = []

    for distance, i in zip(
        distances[0],
        indices[0]
    ):

        result = all_chunks[i].copy()

        result["score"] = float(distance)

        results.append(result)

    return results


# =========================================================
# RAG FUNCTION
# =========================================================

def college_rag(question):

    results = retrieve_documents(
        question,
        k=3
    )

    context_parts = []

    for i, result in enumerate(
        results,
        1
    ):

        source = result["source"]

        if result["page"] is not None:

            location = (
                f"{source}, Page {result['page']}"
            )

        else:

            location = source

        context_parts.append(
            f"[Source {i}: {location}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are CollegeMate, an AI assistant
for college students.

Answer the user's question using ONLY
the context provided below.

Rules:

- Do not invent information.
- Do not use outside knowledge.
- If the answer is not present in the
  context, say:

"I don't have enough information in my
knowledge base to answer that."

- Keep the answer clear and concise.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text, results


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask CollegeMate a question..."
)


if question:

    # Display user message

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching college knowledge base..."
        ):

            answer, sources = college_rag(
                question
            )

        st.markdown(answer)


        # Display sources

        unique_sources = []

        for source in sources:

            if source["source"] not in unique_sources:

                unique_sources.append(
                    source["source"]
                )

        st.markdown("### 📚 Sources")

        for source in unique_sources:

            st.write(
                f"• {source}"
            )


    # Save assistant response

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎓 CollegeMate")

    st.write(
        "RAG-based College Information Assistant"
    )

    st.divider()

    st.write("### 📖 Knowledge Base")

    st.write(
        f"Documents: {len(documents)}"
    )

    st.write(
        f"Chunks: {len(all_chunks)}"
    )

    st.divider()

    st.write("### 💡 Example Questions")

    st.write(
        "• What is the minimum attendance requirement?"
    )

    st.write(
        "• What subjects are taught in Computer Science?"
    )

    st.write(
        "• Who handles placement activities?"
    )

    st.write(
        "• What services does the library provide?"
    )

    st.write(
        "• What happens if attendance is below 75%?"
    )