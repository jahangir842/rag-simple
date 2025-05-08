import chromadb
import requests
from chromadb.utils import embedding_functions
import uuid
import pdfplumber
import os
from pathlib import Path

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
collection_name = "rag_collection"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

# Function to extract text from a PDF CV
def extract_text_from_pdf(pdf_path):
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    text = text.replace("  ", " ")  # Remove double spaces
                    text = "\n".join(line.strip() for line in text.split("\n"))  # Clean line endings
                    text_content.append(text)
                else:
                    text_content.append(f"[No text found on page {page_num}]")
        return " ".join(text_content)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to process all PDFs in the documents directory
def process_documents():
    documents = []
    pdf_files = []
    
    # Get all PDF files from the documents directory
    docs_dir = Path("documents")
    if docs_dir.exists():
        pdf_files = list(docs_dir.glob("*.pdf"))
    
    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            documents.append({
                "text": text,
                "source": pdf_path.name
            })
    
    # Add some general knowledge documents
    general_docs = [
        {
            "text": "The Apollo 11 mission landed humans on the moon for the first time on July 20, 1969.",
            "source": "space_facts"
        },
        {
            "text": "SpaceX's Falcon 9 is a reusable rocket designed to reduce the cost of space travel.",
            "source": "space_facts"
        },
        {
            "text": "The International Space Station orbits Earth at an altitude of about 400 kilometers.",
            "source": "space_facts"
        },
        {
            "text": "Mars Rover Perseverance searches for signs of ancient life on the Martian surface.",
            "source": "space_facts"
        }
    ]
    documents.extend(general_docs)
    return documents

# Store documents in ChromaDB
def store_documents(docs):
    # Get all existing IDs
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
    
    # Prepare documents for storage
    documents_text = [doc["text"] for doc in docs]
    documents_ids = [str(uuid.uuid4()) for _ in docs]
    documents_metadata = [{"source": doc["source"]} for doc in docs]
    
    # Filter out empty documents
    valid_indices = [i for i, doc in enumerate(documents_text) if doc.strip()]
    valid_docs = [documents_text[i] for i in valid_indices]
    valid_ids = [documents_ids[i] for i in valid_indices]
    valid_metadata = [documents_metadata[i] for i in valid_indices]
    
    if valid_docs:
        collection.add(
            documents=valid_docs,
            ids=valid_ids,
            metadatas=valid_metadata
        )
        print(f"Successfully stored {len(valid_docs)} documents in ChromaDB.")
    else:
        print("No valid documents to store.")

# Query ChromaDB for relevant documents
def retrieve_documents(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas"]
    )
    documents = results['documents'][0]
    sources = [meta["source"] for meta in results['metadatas'][0]]
    return documents, sources

def check_llama_server():
    try:
        # Try both the OpenAI-style and llama.cpp native endpoints
        endpoints = [
            "http://localhost:8000/completion",  # llama.cpp native
            "http://localhost:8000/v1/completions"  # OpenAI-style
        ]
        
        for url in endpoints:
            try:
                response = requests.post(
                    url,
                    json={"prompt": "test", "max_tokens": 1},
                    timeout=5
                )
                if response.status_code == 200:
                    return url  # Return the working endpoint
            except requests.exceptions.RequestException:
                continue
                
        return None  # No working endpoint found
    except Exception:
        return None

# Query llama.cpp server with retrieved context
def query_llama(prompt, context, sources):
    # Get the working endpoint
    endpoint = check_llama_server()
    if not endpoint:
        return "Error: Could not connect to llama.cpp server. Please verify the server is running and the endpoint is correct. Try starting the server with: ./server --port 8000"
    
    sources_text = "\nSources: " + ", ".join(sources)
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nBased on the provided context, please answer the question:{sources_text}\nAnswer:"
    
    payload = {
        "prompt": full_prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop": ["\n\n"]  # Stop at double newline to keep responses concise
    }
    
    # If using native endpoint, adjust payload format
    if endpoint.endswith("/completion"):
        payload = {
            "prompt": full_prompt,
            "n_predict": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": ["\n\n"]
        }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        if endpoint.endswith("/completion"):
            return result.get("content", "").strip()
        else:  # OpenAI format
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['text'].strip()
            
        return "Error: Unexpected response format from llama.cpp server"
    except requests.exceptions.RequestException as e:
        return f"Error querying llama.cpp server: {str(e)}"
    except ValueError as e:
        return f"Error parsing server response: {str(e)}"

# Main RAG pipeline
def rag_pipeline(query):
    # Retrieve relevant documents
    retrieved_docs, sources = retrieve_documents(query)
    context = " ".join(retrieved_docs)
    # Query llama.cpp with context
    answer = query_llama(query, context, sources)
    return answer

if __name__ == "__main__":
    print("Starting RAG application...")
    
    # Check if llama.cpp server is running
    if not check_llama_server():
        print("\nWarning: llama.cpp server is not running!")
        print("Please start the server using llama.cpp with the following command:")
        print("./server -m <path-to-your-model> -c 2048")
        print("\nContinuing with document processing...\n")
    
    print("Processing documents...")
    
    # Process and store documents
    documents = process_documents()
    store_documents(documents)
    
    print("\nSystem is ready for queries!")
    print("Example queries you can try:")
    print("1. What are the professional skills mentioned in the CVs?")
    print("2. List the educational qualifications found in the documents.")
    print("3. What are the most common technical skills across all resumes?")
    
    while True:
        try:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nProcessing query...")
            response = rag_pipeline(query)
            print("\nResponse:", response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please make sure the llama.cpp server is running and try again.")