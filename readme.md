Local RAG Vision System with Ollama and Streamlit
This project implements a Retrieval-Augmented Generation (RAG) system with vision capabilities using Ollama locally and Streamlit for the user interface. The system processes PDF documents by manually extracting text, images, and tables, then uses these components for question answering.
Features

Text retrieval using vector embeddings
Image reference tracking and inclusion in responses
Table data integration from CSVs
Local processing with Ollama
Interactive UI with Streamlit

How It Works

Document Processing:

Text is manually extracted from PDFs
Images are manually saved in a dedicated folder
Tables are manually converted to CSV files


Retrieval System:

Text chunks are embedded using Ollama's embedding model
Vector search identifies relevant chunks
References to images and tables are tracked
LLM (via Ollama) generates answers with context and images


User Interface:

Interactive search with Streamlit
Results display with text, images, and tables
Configurable retrieval parameters



Setup Instructions
1. Prerequisites

Python 3.8 or higher
Ollama installed locally (https://ollama.ai)
Required Ollama models:

llava (or another vision model)
nomic-embed-text (or another embedding model)



2. Install Dependencies
bash pip install -r requirements.txt
3. Prepare the Project
bash# Set up the project structure
python setup.py --pdf path/to/your/document.pdf --dir project_folder
This will:

Create the necessary directory structure
Extract text from your PDF (if provided)
Generate a reference guide for manual extraction
Create a sample table CSV

4. Manual Extraction
After running the setup script:

Extract images from the PDF and save them to data/images/ following the naming convention in the reference guide
Extract tables from the PDF and save them as CSVs in data/tables/ following the naming convention

5. Run the Application
bash streamlit run app.py
Project Structure
project_folder/
├── .venv/               # Virtual environment (ignored)
├── data/chroma_db/           # Vector database storage
├── data/images/              # Image documents
├── data/tables/              # Table documents
├── data/text_documents/      # Text documents
├── .env                 # Configuration file
├── app.py               # Main application
├── README.md            # This file
└── requirements.txt 
Customization

Embedding Model: Change EMBEDDING_MODEL in app.py to use a different Ollama embedding model
Vision Model: Change MODEL_NAME in app.py to use a different vision-capable model
Chunk Size: Adjust CHUNK_SIZE and CHUNK_OVERLAP for different text chunking strategies

Limitations

Manual extraction of images and tables is required
Performance depends on the quality of the Ollama models
Limited to the capabilities of the local hardware

Future Enhancements

Add support for automatic image extraction
Improve reference detection with more sophisticated patterns
Add document history and comparison features
Implement caching for improved performance

Troubleshooting

Ollama Connection Issues: Ensure Ollama is running with ollama serve
Image Loading Errors: Check image paths and formats (PNG, JPG, etc.)
Table Processing Errors: Verify CSV format and encoding

bash
rm -rf .venv                            
python3.11 -m venv .venv  
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt