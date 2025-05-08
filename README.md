# RAG-Simple: PDF-Aware Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that combines document storage, semantic search, and language model integration. The system is specifically designed to handle PDF documents (especially CVs/resumes) and general text, providing intelligent question-answering capabilities.

## Features

- 📄 PDF Document Processing: Extract and clean text from PDF files
- 🔍 Semantic Search: Using ChromaDB for efficient document retrieval
- 🤖 LLM Integration: Connects to local llama.cpp server for response generation
- 💾 Persistent Storage: Maintains document embeddings across sessions
- 🎯 CV/Resume Analysis: Specialized handling for resume content

## Prerequisites

- Python 3.8 or higher
- Running llama.cpp server on localhost:8000
- Sufficient disk space for document storage
- PDF documents to analyze (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-simple
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify llama.cpp server:
   ```bash
   curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 10}'
   ```

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── documents/         # Directory for PDF documents
└── chroma_db/         # Vector database storage
```

## Usage

### Adding Documents

1. Place your PDF documents in the `documents/` directory
2. Run the application:
   ```bash
   python app.py
   ```

The system will:
- Extract text from PDF documents
- Generate embeddings using sentence-transformers
- Store documents in ChromaDB for retrieval

### Querying the System

The system supports various types of queries:
- Professional skills extraction from CVs
- General knowledge questions
- Document-specific inquiries

Example usage in Python:

```python
from app import rag_pipeline

# Query about CV
response = rag_pipeline("What are the candidate's technical skills?")
print(response)

# General knowledge query
response = rag_pipeline("Tell me about space exploration.")
print(response)
```

## Configuration

### ChromaDB Settings
- Model: all-MiniLM-L6-v2 (Sentence Transformer)
- Collection: "rag_collection"
- Storage: Persistent storage in ./chroma_db/

### LLM Settings
- Endpoint: http://localhost:8000
- Max Tokens: 200
- Temperature: 0.5

## Technical Details

### RAG Pipeline Flow
1. Document Processing:
   - PDF text extraction with pdfplumber
   - Text cleaning and normalization
   
2. Document Storage:
   - Vector embeddings generation
   - Persistent storage in ChromaDB
   
3. Query Processing:
   - Semantic search for relevant documents
   - Context retrieval (top 3 documents)
   - LLM query with retrieved context

### PDF Processing Features
- Handles multi-page documents
- Removes formatting artifacts
- Maintains text structure
- Error handling for corrupted files

## Troubleshooting

### Common Issues

1. PDF Extraction Fails
   - Ensure PDF is not corrupted
   - Check file permissions
   - Verify PDF is text-based (not scanned)

2. LLM Server Connection
   - Confirm llama.cpp server is running
   - Check localhost:8000 is accessible
   - Verify server has required models loaded

3. ChromaDB Errors
   - Ensure sufficient disk space
   - Check write permissions for chroma_db/
   - Verify embedding model downloads

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ChromaDB team for the vector database
- Sentence-Transformers for embedding generation
- pdfplumber for PDF text extraction
