!pip install PyPDF2
!pip install sentence-transformers
!pip install pinecone-client
!pip install PyPDF2 sentence-transformers faiss-cpu
# Install required libraries if not already installed
# !pip install PyPDF2 sentence-transformers faiss-cpu

# Import necessary libraries
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: PDF Parsing and Data Ingestion
def parse_pdf(file_path):
    """
    Extract text from a PDF file and return it as a list of pages.
    """
    reader = PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages]
    return pages

# Step 2: Text Chunking
def chunk_text(text, max_length=500):
    """
    Split the text into smaller chunks of max_length.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:  # +1 for space
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Step 3: Embedding and Indexing
def create_faiss_index(chunks):
    """
    Create a FAISS index for fast retrieval.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load embedding model
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
    index.add(np.array(embeddings))  # Add embeddings to the index
    return index, embeddings

# Step 4: Query Retrieval
def retrieve_chunks(query, index, chunks, embedding_model, top_k=5):
    """
    Retrieve the most relevant chunks for the query.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Handle empty results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append((chunks[idx], distances[0][i]))
    return results

# Step 5: Response Generation
def generate_response(query, index, chunks, embedding_model):
    """
    Generate a response using retrieved chunks and query.
    """
    relevant_chunks = retrieve_chunks(query, index, chunks, embedding_model)
    if not relevant_chunks:
        return "No relevant information found for your query."

    context = "\n".join([chunk for chunk, _ in relevant_chunks])
    response = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:\n{context}"
    return response

# Main Execution
if __name__ == "__main__":
    # Step 1: Load and Parse PDF
    pdf_path = "/content/drive/MyDrive/ppppp/dt-02.pdf"  # Replace with the path to your PDF file
    pages = parse_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages from the PDF.")

    # Step 2: Process and Chunk Text
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page))
    print(f"Generated {len(all_chunks)} chunks from the PDF.")

    # Step 3: Create FAISS Index
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index, embeddings = create_faiss_index(all_chunks)
    print(f"FAISS index created with {len(embeddings)} embeddings.")

    # Step 4: Example Query
    example_query = "What AI-based approaches are compared in the study for cyberattack detection in cyber-physical systems (CPSs)?"
    response = generate_response(example_query, faiss_index, all_chunks, embedding_model)
    print("\nResponse to Query:")
    print(response)
    #Example Query
    example_query = "ABSTRACT?"
    response = generate_response(example_query, faiss_index, all_chunks, embedding_model)
    print("\nResponse to Query:")
    print(response)
