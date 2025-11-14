import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
import time

# LangChain v1 imports (updated)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Load env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# UI
st.title("GEMINI Model Document Q&A")

st.markdown("""
Welcome to **GEMINI-powered Document Q&A App**!  
📄 Upload any PDF and ask questions.  
⚡ Powered by **Gemini 2.5 Pro** + **FAISS** + **LangChain v1**.
""")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.2
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the given context.

<context>
{context}

Question: {question}
""")

parser = StrOutputParser()

# ---- VECTOR EMBEDDING FUNCTION ----
def vector_embedding():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    if uploaded_files and "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.docs = []

        # Load all PDFs
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(uploaded_file.name)
            st.session_state.docs.extend(loader.load())
            os.remove(uploaded_file.name)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        final_docs = splitter.split_documents(st.session_state.docs)

        st.session_state.final_documents = final_docs
        st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)

# Input box
query = st.text_input("💬 Ask your question:")

# Create vectors
if st.button("🔍 Documents Embedding"):
    if uploaded_files:
        vector_embedding()
        st.success("✅ Vector Store is ready!")
    else:
        st.warning("⚠️ Upload PDFs first.")

# ---- RETRIEVAL + ANSWERING ----
if query:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please run 'Documents Embedding' first.")
    else:
        retriever = st.session_state.vectors.as_retriever()

        # Retrieve relevant docs
        retrieved_docs = retriever.get_relevant_documents(query)

        # Build context text
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Final chain: prompt → LLM → parser
        chain = prompt | llm | parser

        start = time.process_time()
        answer = chain.invoke({"context": context_text, "question": query})

        st.subheader("🧠 Answer:")
        st.write(answer)

        # Show chunks
        with st.expander("📚 Relevant Document Chunks"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"### Chunk {i+1}")
                st.write(doc.page_content)
                st.write("---")
