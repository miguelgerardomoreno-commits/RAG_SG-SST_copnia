"""
crear_embeddings.py
Construye la base vectorial (Chroma) a partir del manual SG-SST en PDF.

ESTRATEGIA DE CHUNKING
----------------------
1. Se limpia cada página (se quitan encabezados/pies de página repetidos).
2. Se concatena TODO el texto del documento en un solo bloque continuo y se
   divide con RecursiveCharacterTextSplitter. Así, el contenido que queda
   partido justo entre el final de una página y el inicio de la siguiente
   no se pierde: el solapamiento (chunk_overlap) cubre ese borde.
3. Cada chunk guarda en sus metadatos la(s) página(s) del PDF con las que
   se solapa, para poder citarlas después.

POR QUÉ CHUNK_SIZE = 450 (y no 1000 como antes)
------------------------------------------------
El modelo de embeddings "paraphrase-multilingual-MiniLM-L12-v2" tiene un
límite de SOLO 128 tokens (~450-500 caracteres en español). Si el chunk es
más largo que eso, el modelo TRUNCA el texto al calcular el embedding: lo
que sobra queda guardado y se le puede mostrar al LLM, pero nunca pudo
"encontrarse" por búsqueda semántica porque el embedding ni siquiera lo
representa. Este es, con alta probabilidad, el motivo de que el bot dijera
"no encontré información" sobre cosas que sí estaban en el manual (cuando
el dato relevante quedaba más allá del carácter ~500 dentro de un chunk
de página completa). Por eso aquí se reduce el chunk a un tamaño que cabe
entero dentro del límite del modelo.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import re
import shutil

PDF_PATH        = "manual_SG-SST.pdf"
DB_DIR          = "db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# El modelo trunca a 128 tokens (~450-500 caracteres en español).
# Se deja margen para no acercarse al límite.
CHUNK_SIZE    = 450
CHUNK_OVERLAP = 90

# ── Limpiar BD anterior ──────────────────────────────────────
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
    print("BD anterior eliminada.")

# ── Cargar PDF ────────────────────────────────────────────────
print("Cargando PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Paginas cargadas: {len(pages)}")

# ── Limpieza de encabezados/pies repetidos ───────────────────
RUIDO = [
    "MANUAL DEL SISTEMA DE GESTIÓN DE",
    "SEGURIDAD Y SALUD EN EL TRABAJO SG-",
    "GH-m-01", "Vigente a partir de",
    "diciembre de 2023", "3ª Actualización",
]
# Antes este filtro buscaba literalmente "de 29" (asumiendo que el PDF
# siempre tiene 29 páginas). Con una expresión regular sigue funcionando
# aunque el manual cambie de tamaño en una futura actualización.
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

pages_limpias = []
for p in pages:
    p.page_content = limpiar(p.page_content)
    if len(p.page_content.strip()) > 80:
        pages_limpias.append(p)

print(f"Paginas con contenido: {len(pages_limpias)}")

# ── Concatenar el documento y registrar los límites de cada página ──
# Esto permite dividir el texto de forma continua (sin perder contenido
# que cae justo en el borde de una página) y, a la vez, saber con qué
# página(s) corresponde cada chunk resultante.
texto_completo = ""
limites_pagina = []  # lista de (inicio_char, fin_char, numero_pagina_1_indexed)
for p in pages_limpias:
    inicio = len(texto_completo)
    texto_completo += p.page_content + "\n\n"
    fin = len(texto_completo)
    numero_pagina = p.metadata.get("page", 0) + 1
    limites_pagina.append((inicio, fin, numero_pagina))

def paginas_de_chunk(inicio, fin):
    """Páginas del PDF con las que se solapa el rango [inicio, fin)."""
    encontradas = {num for (a, b, num) in limites_pagina if inicio < b and fin > a}
    return sorted(encontradas) if encontradas else [1]

# ── Dividir en chunks ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)

chunks_crudos = splitter.create_documents([texto_completo])

chunks = []
for i, ch in enumerate(chunks_crudos):
    inicio = ch.metadata["start_index"]
    fin = inicio + len(ch.page_content)
    paginas = paginas_de_chunk(inicio, fin)
    chunks.append(Document(
        page_content=ch.page_content,
        metadata={
            "page": paginas[0] - 1,   # 0-indexed, por compatibilidad (+1 al mostrar)
            "paginas": paginas,       # todas las páginas con las que se solapa el chunk
            "chunk_id": i,            # id único para deduplicar resultados de búsqueda
        }
    ))

print(f"Total de chunks generados: {len(chunks)}")

longitudes = [len(c.page_content) for c in chunks]
print(f"Tamaño de chunk -> minimo: {min(longitudes)} | maximo: {max(longitudes)} | "
      f"promedio: {sum(longitudes)//len(longitudes)} caracteres")

# ── Verificaciones clave (sanity check) ───────────────────────
# Se comprueba sobre el TEXTO COMPLETO (no por chunk individual), porque
# con chunks pequeños un encabezado y su contenido pueden quedar en
# fragmentos distintos. Lo que sí queremos garantizar aquí es que la
# limpieza de encabezados/pies no haya borrado por accidente texto útil.
secciones_clave = {
    "14.1 Director General":      lambda c: "14.1" in c and "Director General" in c and "Máximo responsable" in c,
    "14.6 COPASST":               lambda c: "14.6" in c or ("COPASST" in c and "Velar" in c),
    "20.5 Promoción y Prev.":     lambda c: "20.5" in c or ("Conservación Visual" in c),
    "Reporte incidentes (GH-pr)": lambda c: "GH-pr-11" in c or ("reporte" in c.lower() and "accidente" in c.lower() and "investigación" in c.lower()),
    "Política SST (sección 4)":   lambda c: "se compromete con" in c.lower() and "copnia" in c.lower(),
}

print("\n-- Verificando que la limpieza no haya borrado secciones clave --")
for nombre, check in secciones_clave.items():
    encontrado = check(texto_completo)
    estado = "OK" if encontrado else "FALTA"
    print(f"  [{estado}] {nombre}")

# ── Embeddings + base vectorial ───────────────────────────────
print("\nCargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Creando base vectorial...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
    ids=[str(c.metadata["chunk_id"]) for c in chunks],
)

print(f"BD creada con {vector_db._collection.count()} vectores")
print("Listo. Ejecuta: python chat_rag.py   (o streamlit run app.py)")