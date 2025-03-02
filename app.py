# Streamlit-based Multimodal RAG powered by Qwen 2.5 VL
import os
import base64
import gc
import tempfile
import uuid
import time
import streamlit as st
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image
from embedding import EmbedData
from vector_db import QdrantVDB_QB, Retriever
from rag import RAG

# Define collection name
collection_name = "multimodal_rag_collection"

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def reset_chat():
    """Clears chat history and garbage collects memory"""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    """Displays PDF preview in Streamlit"""
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# ✅ Load and encode image from Google Drive to Base64
image_path = "/content/drive/MyDrive/multimodal-rag/qwen2.5vl_logo.png"
try:
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    img_html = f'<img src="data:image/png;base64,{img_base64}" width="220" style="vertical-align: -0px;">'
except Exception as e:
    img_html = "<p style='color:red'>⚠️ Logo Image Not Found</p>"

# Sidebar file uploader
with st.sidebar:
    st.header("📄 Add Your Document")
    uploaded_file = st.file_uploader("Choose a `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("🔄 Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):
                    # Convert PDF pages to images
                    images = convert_from_path(file_path)
                    image_paths = []
                    os.makedirs("./images", exist_ok=True)

                    for i, img in enumerate(images):
                        img_path = f"./images/page{i}.jpg"
                        img.save(img_path, "JPEG")
                        image_paths.append(img_path)

                    # Embed images
                    embeddata = EmbedData()
                    embeddata.embed(image_paths)

                    # Set up Qdrant vector database
                    qdrant_vdb = QdrantVDB_QB(collection_name=collection_name, vector_dim=128)
                    qdrant_vdb.define_client()
                    qdrant_vdb.create_collection()
                    qdrant_vdb.ingest_data(embeddata)

                    # Set up retriever and RAG system
                    retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
                    query_engine = RAG(retriever=retriever)

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the document is ready
                st.success("✅ Ready to Chat!")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            st.stop()

# Chat interface
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown(f"""
    # Multimodal RAG powered by {img_html}
    """, unsafe_allow_html=True)

with col2:
    st.button("Clear Chat ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the RAG system
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streaming_response = query_engine.query(prompt)
                
        for chunk in streaming_response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        
        message_placeholder.markdown(full_response)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
