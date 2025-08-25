import os
import streamlit as st
import pickle
import time
import tempfile 
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

#environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing! Please add it in the .env file.")

# CSS Styling
st.markdown(
    """
    <style>
    /* Futuristic Title */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 28px;
        text-shadow: 0 0 10px #00f2fe, 0 0 20px #4facfe;
        letter-spacing: 1px;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00f2fe, 0 0 20px #4facfe; }
        to { text-shadow: 0 0 20px #00f2fe, 0 0 40px #4facfe; }
    }

    /* Sidebar - Glassmorphism Effect */
    section[data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        color: #ffffff;
        font-weight: 600;
        padding: 12px;
    }

    /* Chat/Response Area */
    .stMarkdown {
        font-size: 16px;
        line-height: 1.7;
        color: #e0e0e0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
        transition: 0.3s;
    }

    .stMarkdown:hover {
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main App Title
st.markdown('<div class="main-title">InsightBot: Token-Efficient Research 🤖</div>', unsafe_allow_html=True)
st.sidebar.markdown("Input Sources")

# If old FAISS index exists, remove it before creating a new one
if os.path.exists("faiss_index.pkl"):
    os.remove("faiss_index.pkl")
    
# URLs input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# PDF input
pdf_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

process_clicked = st.sidebar.button("Process Inputs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)
if process_clicked:
    all_docs = []

    #Process URLs
    valid_urls = [u for u in urls if u.strip()]
    if valid_urls:
        main_placeholder.text("Data Loading from URLs...✅")
        for link in valid_urls:
            try:
                loader = WebBaseLoader([link])   # expects a list, not `urls=`
                url_docs = loader.load()
                all_docs.extend(url_docs)
            except Exception as e:
                st.warning(f"Could not load {link}: {e}")


    #Process PDFs
    if pdf_files:
        main_placeholder.text("Data Loading from PDFs...✅")
        for pdf in pdf_files:
            # Saving uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.read())
                tmp_path = tmp_file.name

            # Loading PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            pdf_docs = loader.load()
            all_docs.extend(pdf_docs)

    # Checking if any documents were loaded
    if not all_docs:
        st.error("❌ No data could be loaded from the provided inputs. Please check your URLs or PDFs.")
    else:
        # Splitting into chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200,
        )
        main_placeholder.text("Text Splitter...✅")
        docs = text_splitter.split_documents(all_docs)

        #embeddings and FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Saving vectorstore
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        main_placeholder.text("Embedding Vector Store Rebuilt...✅")
        time.sleep(2)


# query
query = st.text_input("Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
