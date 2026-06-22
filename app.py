import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

DB_DIR          = "db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"

st.set_page_config(page_title="SSTBot - Asistente SG-SST", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main { background-color: #f4f6f8; }
h1 { color: #1F3A5F; }
.stTextInput > div > div > input { border-radius: 8px; }
.stButton>button {
    background-color: #1F3A5F; color: white;
    border-radius: 8px; border: none;
    height: 3em; width: 100%; font-size: 16px;
}
.stButton>button:hover { background-color: #2E5A88; }
.msg-bot {
    background-color: white; padding: 16px 20px;
    border-radius: 10px; border-left: 5px solid #1F3A5F;
    box-shadow: 0px 1px 6px rgba(0,0,0,0.08); margin-bottom: 12px;
}
.msg-user {
    background-color: #e8eef5; padding: 12px 18px;
    border-radius: 10px; border-right: 5px solid #2E5A88;
    text-align: right; margin-bottom: 8px;
}
.paginas { font-size: 0.82em; color: #555; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

load_dotenv()

@st.cache_resource(show_spinner="Cargando sistema RAG...")
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # k sube respecto a la versión anterior porque ahora los chunks son
    # más pequeños (para respetar el límite de 128 tokens del modelo de
    # embeddings), así que se necesitan más fragmentos para cubrir el
    # mismo contenido.
    retriever_sim = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )
    retriever_mmr = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.5}
    )
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0, max_tokens=1500)
    return retriever_sim, retriever_mmr, llm

retriever_sim, retriever_mmr, llm = cargar_sistema()

def recuperar_docs(pregunta):
    docs_sim = retriever_sim.invoke(pregunta)
    docs_mmr = retriever_mmr.invoke(pregunta)
    vistos = set()
    combinados = []
    for doc in docs_sim + docs_mmr:
        # Deduplicación por id único del chunk. Antes se usaba el inicio
        # del texto, lo que descartaba por error chunks distintos que
        # compartían las mismas primeras palabras (perdiendo contenido real).
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

st.title("🛡️ SSTBot")
st.subheader("Asistente especializado en el Manual SG-SST")
st.markdown("---")

if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje in st.session_state.historial:
    if mensaje["rol"] == "usuario":
        st.markdown(f"<div class='msg-user'>🧑 {mensaje['texto']}</div>", unsafe_allow_html=True)
    else:
        paginas_str = f"📄 Páginas consultadas: {mensaje['paginas']}" if mensaje.get("paginas") else ""
        st.markdown(
            f"<div class='msg-bot'>🛡️ {mensaje['texto']}<div class='paginas'>{paginas_str}</div></div>",
            unsafe_allow_html=True
        )

col1, col2 = st.columns([5, 1])
with col1:
    pregunta = st.text_input(
        "", placeholder="Ej: ¿Cuáles son las responsabilidades del Director General?",
        label_visibility="collapsed"
    )
with col2:
    consultar = st.button("Consultar")

if st.session_state.historial:
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.historial = []
        st.rerun()

if consultar and pregunta.strip():
    with st.spinner("Buscando en el manual..."):
        documentos = recuperar_docs(pregunta)

        contexto = ""
        for doc in documentos:
            paginas_chunk = doc.metadata.get("paginas", [doc.metadata.get("page", 0) + 1])
            tipo  = doc.metadata.get("chunk_type", "")
            label = formato_paginas(paginas_chunk).capitalize() + (f" [{tipo}]" if tipo else "")
            contexto += f"\n[{label}]\n{doc.page_content}\n"

        paginas = sorted(set(
            p for doc in documentos
            for p in doc.metadata.get("paginas", [doc.metadata.get("page", 0) + 1])
        ))

        ultimos = st.session_state.historial[-6:]
        historial_texto = "\n".join(
            f"{'Usuario' if m['rol']=='usuario' else 'SSTBot'}: {m['texto']}"
            for m in ultimos
        ) or "Sin historial previo."

        prompt_final = prompt.format(
            contexto=contexto, pregunta=pregunta, historial=historial_texto
        )
        respuesta = llm.invoke(prompt_final)
        texto_respuesta = respuesta.content

    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
    st.session_state.historial.append({
        "rol": "bot", "texto": texto_respuesta, "paginas": paginas
    })
    st.rerun()

elif consultar and not pregunta.strip():
    st.warning("Por favor ingresa una pregunta.")