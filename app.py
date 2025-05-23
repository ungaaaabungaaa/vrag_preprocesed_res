import os
import re
import pandas as pd
from PIL import Image
import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests  # For Ollama API calls
import logging
import torch

# Force CPU as the only device
DEVICE = torch.device("cpu")
torch.set_default_device('cpu')  # Ensure all tensors are created on CPU

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_FOLDER = "data/text_documents"
IMAGE_FOLDER = "data/images"
TABLE_FOLDER = "data/tables"
DB_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:7b"  # Default to mistral 7b
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint

@st.cache_resource
def load_models():
    try:
        # Load sentence transformer with CPU explicitly
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Load BLIP processor and model with CPU explicitly
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float32)
        blip_model = blip_model.to('cpu')
        blip_model.eval()

        return embedding_model, processor, blip_model

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise e

# Initialize ChromaDB - UPDATED for new ChromaDB API
@st.cache_resource
def init_db():
    # Ensure the DB directory exists
    os.makedirs(DB_PATH, exist_ok=True)

    try:
        # Using the updated ChromaDB client initialization
        client = PersistentClient(path=DB_PATH)

        try:
            collection = client.get_collection("documents")
            logger.info(f"Found existing collection with {collection.count()} documents")
        except Exception as e:
            logger.info(f"Creating new collection: {str(e)}")
            collection = client.create_collection("documents")

        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

# Function to call Ollama API
def generate_with_ollama(prompt, model=OLLAMA_MODEL, max_tokens=500):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
        return "Sorry, I couldn't generate a response. Please check if Ollama is running."

# Modified response generation function to use Ollama
def generate_response(query, collection, embedding_model):
    context = retrieve_context(collection, embedding_model, query)

    # Build prompt with all context types
    prompt = f"""Answer the following question based on the provided context.
When referencing figures or tables, use the format [Figure X] or [Table Y].

Question: {query}

Text Context: {' '.join(context['text'])}

Image Context: {' '.join([img['caption'] for img in context['images']])}

Table Context: {' '.join([tbl['description'] for tbl in context['tables']])}

Answer:"""

    response = generate_with_ollama(prompt)
    return response, context

# Function to retrieve context from the database
def retrieve_context(collection, embedding_model, query, n_results=3):
    try:
        # Get embeddings for the query
        query_embedding = embedding_model.encode(query).tolist()

        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Process the results into context buckets
        context = {
            "text": [],
            "images": [],
            "tables": []
        }

        if results and 'documents' in results and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if 'metadatas' in results and len(results['metadatas']) > 0 else {}
                doc_type = metadata.get('type', 'text')

                if doc_type == 'text':
                    context['text'].append(doc)
                elif doc_type == 'image':
                    context['images'].append({
                        'caption': doc,
                        'path': metadata.get('path', '')
                    })
                elif doc_type == 'table':
                    context['tables'].append({
                        'description': doc,
                        'path': metadata.get('path', '')
                    })

        return context
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return {"text": [], "images": [], "tables": []}

# Display the response and context in the UI
def display_response(response, context):
    st.markdown("### Answer")
    st.write(response)

    # Display referenced context if present
    if any(context.values()):
        with st.expander("View Referenced Context"):
            if context['text']:
                st.markdown("#### Text References")
                for i, text in enumerate(context['text']):
                    st.markdown(f"**Text {i+1}**: {text[:300]}...")

            if context['images']:
                st.markdown("#### Image References")
                cols = st.columns(min(3, len(context['images'])))
                for i, img in enumerate(context['images']):
                    with cols[i % len(cols)]:
                        try:
                            if os.path.exists(img['path']):
                                st.image(img['path'], caption=f"Figure {i+1}")
                                st.write(img['caption'])
                            else:
                                st.error(f"Image not found: {img['path']}")
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")

            if context['tables']:
                st.markdown("#### Table References")
                for i, tbl in enumerate(context['tables']):
                    st.markdown(f"**Table {i+1}**: {tbl['description']}")
                    try:
                        if os.path.exists(tbl['path']):
                            df = pd.read_csv(tbl['path'])
                            st.dataframe(df)
                        else:
                            st.error(f"Table not found: {tbl['path']}")
                    except Exception as e:
                        st.error(f"Error displaying table: {str(e)}")

# Ensure folder exists
def ensure_folders_exist():
    folders = [TEXT_FOLDER, IMAGE_FOLDER, TABLE_FOLDER, DB_PATH]
    existing = []
    missing = []

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            missing.append(folder)
        else:
            existing.append(folder)

    return existing, missing

# Function to clear database
def clear_database(collection):
    try:
        # Get all document IDs and delete them
        all_docs = collection.get()
        if all_docs and 'ids' in all_docs and all_docs['ids']:
            collection.delete(ids=all_docs['ids'])
            return len(all_docs['ids'])
        return 0
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return 0

# Index documents to populate the database
def index_documents(collection, embedding_model, processor, blip_model):
    st.info("Starting document indexing...")

    # Check and create folders
    existing, created = ensure_folders_exist()
    if created:
        st.info(f"Created missing folders: {', '.join(created)}")

    # Count successful indexes
    text_count = index_text_documents(collection, embedding_model)
    image_count = index_images(collection, embedding_model, processor, blip_model)
    table_count = index_tables(collection, embedding_model)

    return text_count, image_count, table_count

# Index text documents
def index_text_documents(collection, embedding_model):
    count = 0
    if not os.path.exists(TEXT_FOLDER):
        st.warning(f"Text folder {TEXT_FOLDER} not found.")
        return count

    files = os.listdir(TEXT_FOLDER)
    if not files:
        st.info(f"No text files found in {TEXT_FOLDER}")
        return count

    for filename in files:
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(TEXT_FOLDER, filename)
            try:
                # Try UTF-8 first, then fall back to other encodings if that fails
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='cp1252') as f:  # Windows-1252
                            content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='iso-8859-1') as f:  # Latin-1
                            content = f.read()

                # Skip empty files
                if not content.strip():
                    continue

                # Split content into chunks (simple approach)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

                for i, chunk in enumerate(chunks):
                    # Skip empty chunks
                    if not chunk.strip():
                        continue

                    try:
                        embedding = embedding_model.encode(chunk).tolist()

                        # Add to collection
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            metadatas=[{
                                'type': 'text',
                                'source': filename,
                                'chunk': i
                            }],
                            ids=[f"txt_{filename}_{i}"]
                        )
                        count += 1
                    except Exception as e:
                        st.warning(f"Error processing chunk {i} of {filename}: {str(e)}")
                        continue

            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")

    return count

# Index images with captions
def index_images(collection, embedding_model, processor, blip_model):
    count = 0
    if not os.path.exists(IMAGE_FOLDER):
        st.warning(f"Image folder {IMAGE_FOLDER} not found.")
        return count

    files = os.listdir(IMAGE_FOLDER)
    if not files:
        st.info(f"No image files found in {IMAGE_FOLDER}")
        return count

    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                # Generate caption
                image = Image.open(file_path).convert('RGB')
                
                # Explicitly use CPU for processing
                inputs = processor(image, return_tensors="pt").to('cpu')

                with torch.no_grad():
                    outputs = blip_model.generate(**inputs)

                caption = processor.decode(outputs[0], skip_special_tokens=True)

                # Get embedding for the caption
                embedding = embedding_model.encode(caption).tolist()

                # Add to collection
                collection.add(
                    documents=[caption],
                    embeddings=[embedding],
                    metadatas=[{
                        'type': 'image',
                        'path': file_path,
                        'source': filename
                    }],
                    ids=[f"img_{filename}"]
                )
                count += 1
            except Exception as e:
                st.error(f"Error processing image {filename}: {str(e)}")
    return count

# Index tables with descriptions
def index_tables(collection, embedding_model):
    count = 0
    if not os.path.exists(TABLE_FOLDER):
        st.warning(f"Table folder {TABLE_FOLDER} not found.")
        return count

    files = os.listdir(TABLE_FOLDER)
    if not files:
        st.info(f"No table files found in {TABLE_FOLDER}")
        return count

    for filename in files:
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(TABLE_FOLDER, filename)
            try:
                # Read table
                df = pd.read_csv(file_path)

                # Generate description
                num_rows, num_cols = df.shape
                columns = ", ".join(df.columns.tolist())
                description = f"Table {filename} with {num_rows} rows and {num_cols} columns. Columns: {columns}"

                # Sample data
                sample_data = str(df.head(3))
                full_text = f"{description}\n\nSample data:\n{sample_data}"

                # Get embedding
                embedding = embedding_model.encode(full_text).tolist()

                # Add to collection
                collection.add(
                    documents=[full_text],
                    embeddings=[embedding],
                    metadatas=[{
                        'type': 'table',
                        'path': file_path,
                        'source': filename
                    }],
                    ids=[f"tbl_{filename}"]
                )
                count += 1
            except Exception as e:
                st.error(f"Error processing table {filename}: {str(e)}")
    return count

# File upload functionality
def handle_file_uploads():
    st.header("Upload Documents")

    # Text file upload
    text_files = st.file_uploader("Upload Text Files", type=["txt"], accept_multiple_files=True)
    if text_files:
        ensure_folders_exist()  # Make sure folders exist before saving
        for uploaded_file in text_files:
            save_path = os.path.join(TEXT_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(text_files)} text files")

    # Image file upload
    image_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp", "gif"], accept_multiple_files=True)
    if image_files:
        ensure_folders_exist()  # Make sure folders exist before saving
        for uploaded_file in image_files:
            save_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(image_files)} image files")

    # Table file upload
    table_files = st.file_uploader("Upload Tables", type=["csv"], accept_multiple_files=True)
    if table_files:
        ensure_folders_exist()  # Make sure folders exist before saving
        for uploaded_file in table_files:
            save_path = os.path.join(TABLE_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(table_files)} table files")

# Check Ollama connection
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            return True, available_models
        else:
            return False, []
    except requests.exceptions.RequestException:
        return False, []

# Main Streamlit app
def main():
    st.title("Multi-Modal RAG Chatbot with Ollama")
    st.write("This chatbot can process and reference text, images, and tables from your documents.")

    # Display device information
    st.info(f"Running on: CPU (CUDA explicitly disabled)")

    # Initialize everything
    try:
        embedding_model, processor, blip_model = load_models()
        collection = init_db()

        # Sidebar for document management
        with st.sidebar:
            st.header("Document Management")

            # Add file upload functionality to sidebar
            handle_file_uploads()

            st.markdown("---")

            # Database management buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Re-index Documents", type="primary"):
                    with st.spinner("Indexing documents..."):
                        try:
                            text_count, image_count, table_count = index_documents(collection, embedding_model, processor, blip_model)
                            st.success(f"Documents indexed! Added {text_count} text chunks, {image_count} images, and {table_count} tables.")
                        except Exception as e:
                            st.error(f"Error during indexing: {str(e)}")

            with col2:
                if st.button("Clear Database", type="secondary"):
                    with st.spinner("Clearing database..."):
                        try:
                            cleared_count = clear_database(collection)
                            st.success(f"Database cleared! Removed {cleared_count} documents.")
                        except Exception as e:
                            st.error(f"Error clearing database: {str(e)}")

            # Check and display DB status
            try:
                count = collection.count()
                st.markdown("### Database Info")
                st.write(f"Documents in DB: {count}")
            except Exception as e:
                st.error(f"Error getting DB count: {str(e)}")

            # Ollama status check
            st.markdown("### Ollama Settings")
            st.write(f"Current model: {OLLAMA_MODEL}")

            ollama_connected, available_models = check_ollama_status()
            if ollama_connected:
                st.success("Ollama status: Connected")
                if available_models:
                    with st.expander("Available models"):
                        for model in available_models:
                            st.write(f"• {model}")
                else:
                    st.warning("No models found. Please pull a model first.")
            else:
                st.error("Ollama status: Not connected. Make sure Ollama is running.")
                st.info("To start Ollama, run: `ollama serve`")

        # Main chat interface
        query = st.text_input("Ask your question:", key="query")

        if query:
            # Check if we have documents in the database
            try:
                doc_count = collection.count()
                if doc_count == 0:
                    st.warning("No documents found in the database. Please upload and index some documents first.")
                else:
                    # Check Ollama connection before proceeding
                    ollama_connected, _ = check_ollama_status()
                    if not ollama_connected:
                        st.error("Cannot generate response: Ollama is not connected. Please make sure Ollama is running.")
                    else:
                        with st.spinner("Generating response..."):
                            response, context = generate_response(query, collection, embedding_model)
                        display_response(response, context)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

    except Exception as e:
        st.error(f"Critical error in application: {str(e)}")
        st.info("Troubleshooting tips:")
        st.write("1. Make sure all dependencies are installed: `pip install -r requirements.txt`")
        st.write("2. Check that Ollama is running: `ollama serve`")
        st.write("3. Make sure you have pulled the mistral model: `ollama pull mistral:7b`")
        st.write("4. Verify that you have the necessary disk space for the embeddings model and BLIP model.")
        st.write("5. Try restarting the Streamlit application")


if __name__ == "__main__":
    # Set environment variables to force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide any CUDA devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging
    main()