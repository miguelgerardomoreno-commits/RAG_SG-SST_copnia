from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

DB_DIR          = "db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"

load_dotenv()

print("⏳ Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# Como ahora los chunks son más pequeños (para respetar el límite de 128
# tokens del modelo de embeddings), se necesitan más fragmentos para
# cubrir el mismo contenido. Por eso k sube respecto a la versión anterior.
retriever_sim = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)
retriever_mmr = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.5}
)

def recuperar_docs(pregunta):
    """Combina resultados de similarity + MMR, deduplica por chunk_id."""
    docs_sim = retriever_sim.invoke(pregunta)
    docs_mmr = retriever_mmr.invoke(pregunta)
    vistos = set()
    combinados = []
    for doc in docs_sim + docs_mmr:
        # Antes se deduplicaba con los primeros 200 caracteres del chunk.
        # Eso descartaba por error chunks distintos que compartían el
        # mismo inicio de texto (perdiendo contenido real). Ahora se usa
        # el id único asignado a cada chunk al crear la base vectorial.
        clave = doc.metadata.get("chunk_id", doc.page_content)
        if clave not in vistos:
            vistos.add(clave)
            combinados.append(doc)
    return combinados

def formato_paginas(lista_paginas):
    paginas = sorted(set(lista_paginas))
    if len(paginas) == 1:
        return f"página {paginas[0]}"
    return f"páginas {', '.join(str(p) for p in paginas)}"

llm = ChatGroq(
    model_name=LLM_MODEL,
    temperature=0,
    max_tokens=1500
)

prompt = PromptTemplate(
    input_variables=["contexto", "pregunta", "historial"],
    template="""Eres "SSTBot", asistente experto en el Manual SG-SST del COPNIA.

INSTRUCCIONES:
1. Lee TODOS los fragmentos del CONTEXTO antes de responder.
2. Cuando una sección tenga un encabezado numerado (ej: "14.1 Director General",
   "14.6 COPASST", "20.5 Programa de Promoción"), las viñetas (•) que le siguen
   son LAS RESPONSABILIDADES O FUNCIONES DE ESA SECCIÓN ESPECÍFICA.
   No las mezcles con otras secciones del contexto.
3. Responde con la información de la sección que más directamente responda la pregunta.
4. Si hay viñetas (•) relevantes, inclúyelas TODAS en tu respuesta.
5. No uses conocimiento externo ni inventes información.
6. Si ningún fragmento toca el tema, responde exactamente:
   "No encontré información sobre esa pregunta en el manual SG-SST."

HISTORIAL:
{historial}

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""
)

historial = []

print("\n" + "=" * 60)
print("  🛡️  SSTBot - Asistente SG-SST (modo consola)")
print("  Escribe 'salir' para finalizar | 'limpiar' para reiniciar")
print("=" * 60)

while True:
    pregunta = input("\n🧑 Pregunta: ").strip()

    if not pregunta:
        continue
    if pregunta.lower() == "salir":
        print("\n👋 Hasta pronto.")
        break
    if pregunta.lower() == "limpiar":
        historial = []
        print("✓ Historial limpiado.")
        continue

    documentos = recuperar_docs(pregunta)

    contexto = ""
    for doc in documentos:
        paginas_chunk = doc.metadata.get("paginas", [doc.metadata.get("page", 0) + 1])
        tipo  = doc.metadata.get("chunk_type", "")
        label = formato_paginas(paginas_chunk).capitalize() + (f" [{tipo}]" if tipo else "")
        contexto += f"\n[{label}]\n{doc.page_content}\n"

    todas_paginas = sorted(set(
        p for doc in documentos
        for p in doc.metadata.get("paginas", [doc.metadata.get("page", 0) + 1])
    ))

    ultimos = historial[-6:]
    historial_texto = "\n".join(
        f"{'Usuario' if m['rol']=='usuario' else 'SSTBot'}: {m['texto']}"
        for m in ultimos
    ) or "Sin historial previo."

    prompt_final = prompt.format(
        contexto=contexto,
        pregunta=pregunta,
        historial=historial_texto
    )

    respuesta = llm.invoke(prompt_final)
    texto = respuesta.content

    historial.append({"rol": "usuario", "texto": pregunta})
    historial.append({"rol": "bot",     "texto": texto})

    print(f"\n🛡️  SSTBot:\n{texto}")
    print(f"\n📄 Páginas consultadas: {todas_paginas}")
    print("-" * 60)