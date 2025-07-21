import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# App title
st.title("GEMMA Model Document Q&A")

st.markdown("""
Welcome to **GEMMA-powered Document Q&A App**!  
📄 Upload any PDF documents, and 🤖 ask questions about their content.  
This app uses **Gemma 2-9B Instruct Model via Groq API** to give you fast and accurate answers.

**How it works:**
1. Upload one or more PDF files.
2. Click on **"Documents Embedding"** to process the content.
3. Ask any question related to the uploaded documents.
4. Get instant answers with the most relevant document excerpts!

> Built with ❤️ using Streamlit, LangChain, Groq, and Google Embeddings.
""")
    


# Upload PDF files
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize GEMMA LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided only.
    <context>
    {context}
    Question: {input}
    """
)

# Document Embedding Function
def vector_embedding():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    if uploaded_files and "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
        st.session_state.docs = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            st.session_state.docs.extend(loader.load())
            os.remove(uploaded_file.name)  # Optional cleanup

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# User Input
prompt1 = st.text_input("💬 What do you want to ask from the uploaded documents?")

# Button to trigger vector creation
if st.button("🔍 Documents Embedding"):
    if uploaded_files:
        vector_embedding()
        st.success("✅ Vector Store DB is ready!")
    else:
        st.warning("⚠️ Please upload PDF files first.")

# Q&A Interaction
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click 'Documents Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt1})
        st.subheader("🧠 Answer:")
        st.write(response['answer'])

        with st.expander("📚 Document Context (Most Relevant Chunks)"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
