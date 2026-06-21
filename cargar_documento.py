from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar PDF
loader = PyPDFLoader("manual_SG-SST.pdf")
documentos = loader.load()

print(f"Total páginas: {len(documentos)}")

# Crear chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documentos)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}")
    print(chunk.page_content[:300])