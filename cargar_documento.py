"""
cargar_documento.py
Script de PREVISUALIZACIÓN: carga el PDF, limpia el texto y genera los
mismos chunks que va a usar crear_embeddings.py, sin tocar la base
vectorial. Sirve para revisar rápido cómo van a quedar los fragmentos
antes de reconstruir la base de datos completa.

IMPORTANTE: estos parámetros deben coincidir siempre con los de
crear_embeddings.py (CHUNK_SIZE, CHUNK_OVERLAP y la función de limpieza),
o esta vista previa no representará lo que realmente queda indexado.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

PDF_PATH      = "manual_SG-SST.pdf"
CHUNK_SIZE    = 450
CHUNK_OVERLAP = 90

RUIDO = [
    "MANUAL DEL SISTEMA DE GESTIÓN DE",
    "SEGURIDAD Y SALUD EN EL TRABAJO SG-",
    "GH-m-01", "Vigente a partir de",
    "diciembre de 2023", "3ª Actualización",
]
PATRON_PIE_PAGINA = re.compile(r"^Página\s+\d+\s+de\s+\d+", re.IGNORECASE)

def limpiar(texto):
    lineas = []
    for linea in texto.split("\n"):
        s = linea.strip()
        if any(s.startswith(r) for r in RUIDO):
            continue
        if PATRON_PIE_PAGINA.match(s):
            continue
        lineas.append(linea)
    return "\n".join(lineas).strip()

# ── Cargar PDF ────────────────────────────────────────────────
print(f"Cargando: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

pages_limpias = []
for p in pages:
    p.page_content = limpiar(p.page_content)
    if len(p.page_content.strip()) > 80:
        pages_limpias.append(p)

print(f"Paginas con contenido: {len(pages_limpias)}")

# ── Concatenar y registrar límites de página (igual que crear_embeddings.py) ──
texto_completo = ""
limites_pagina = []
for p in pages_limpias:
    inicio = len(texto_completo)
    texto_completo += p.page_content + "\n\n"
    fin = len(texto_completo)
    numero_pagina = p.metadata.get("page", 0) + 1
    limites_pagina.append((inicio, fin, numero_pagina))

def paginas_de_chunk(inicio, fin):
    encontradas = {num for (a, b, num) in limites_pagina if inicio < b and fin > a}
    return sorted(encontradas) if encontradas else [1]

# ── Crear chunks ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)

chunks = splitter.create_documents([texto_completo])
print(f"Total chunks: {len(chunks)}")

# ── Estadísticas ───────────────────────────────────────────────
longitudes = [len(c.page_content) for c in chunks]
print("\nEstadisticas de chunks:")
print(f"   Minimo  : {min(longitudes)} chars")
print(f"   Maximo  : {max(longitudes)} chars")
print(f"   Promedio: {sum(longitudes)//len(longitudes)} chars")

# ── Vista previa ───────────────────────────────────────────────
print("\n-- Vista previa de los primeros 3 chunks --")
for i, ch in enumerate(chunks[:3]):
    inicio = ch.metadata["start_index"]
    fin = inicio + len(ch.page_content)
    paginas = paginas_de_chunk(inicio, fin)
    print(f"\nChunk {i+1} | Pagina(s) {paginas} | {len(ch.page_content)} chars")
    print(ch.page_content[:400])
    print("...")