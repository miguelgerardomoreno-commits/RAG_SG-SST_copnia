from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar PDF
loader = PyPDFLoader("manual_SG-SST.pdf")
documents = loader.load()

# Crear chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks creados: {len(chunks)}")

# Embeddings locales
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Crear BD vectorial
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

vector_db.persist()

print("Base vectorial creada correctamente.")