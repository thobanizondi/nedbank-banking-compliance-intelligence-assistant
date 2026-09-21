import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.getLogger("pypdf").setLevel(logging.ERROR)

# Load .env from the project root, regardless of where this script is run from
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

# --- Azure AI Search: connectivity check ---
search_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_key))
indexes = list(search_client.list_indexes())
print(f"Connected to Azure AI Search. Existing indexes: {[idx.name for idx in indexes]}")

# --- Azure Blob Storage: list and download documents ---
conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(conn_str)
container_client = blob_service_client.get_container_client(container_name)

# Store downloaded docs in the project root, not inside ingestion/
download_folder = os.path.join(os.path.dirname(__file__), "..", "downloaded_docs")
os.makedirs(download_folder, exist_ok=True)

blob_list = list(container_client.list_blobs())
if not blob_list:
    print(f"No files found in container '{container_name}'. Upload some documents first.")
else:
    for blob in blob_list:
        print(f"Found blob: {blob.name}")
        download_path = os.path.join(download_folder, blob.name)
        with open(download_path, "wb") as f:
            f.write(container_client.download_blob(blob.name).readall())
        print(f"Downloaded to: {download_path}")
    print(f"\nDone. {len(blob_list)} file(s) downloaded to '{download_folder}/'.")

# --- Load and chunk documents ---
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

for filename in os.listdir(download_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(download_folder, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"{filename}: {len(pages)} pages -> {len(chunks)} chunks")

print(f"\nTotal chunks across all documents: {len(all_chunks)}")

# --- Generate embeddings and push chunks to Azure AI Search ---
print("\nLoading embedding model (this may take a moment on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key,
    index_name=search_index_name,
    embedding_function=embeddings.embed_query,
)

print(f"Pushing {len(all_chunks)} chunks to Azure AI Search index '{search_index_name}'...")
vector_store.add_documents(documents=all_chunks)
print("Done. Chunks indexed successfully.")