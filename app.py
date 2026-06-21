import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# -----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="SSTBot - Asistente SG-SST",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# ESTILOS CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f4f6f8;
}

h1 {
    color: #1F3A5F;
}

.stTextInput > div > div > input {
    border-radius: 8px;
}

.stButton>button {
    background-color: #1F3A5F;
    color: white;
    border-radius: 8px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #2E5A88;
    color: white;
}

.respuesta {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1F3A5F;
    box-shadow: 0px 1px 6px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# CARGAR VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# CARGAR MODELOS
# -----------------------------
@st.cache_resource
def cargar_sistema():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm


retriever, llm = cargar_sistema()

# -----------------------------
# PROMPT
# -----------------------------
prompt = PromptTemplate(
    input_variables=["contexto", "pregunta"],
    template="""
Eres "SSTBot", asistente especializado en el Manual del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST) del COPNIA.

Reglas:

1. Responde únicamente utilizando el contexto proporcionado.
2. No inventes información.
3. Analiza cuidadosamente todo el contexto antes de responder.
4. Si encuentras información parcialmente relacionada, úsala para responder.
5. Solo responde:

"No encontré información sobre esa pregunta en el manual SG-SST."

si después de revisar completamente el contexto no existe información relevante.

6. Sé claro y profesional.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""
)

# -----------------------------
# CABECERA
# -----------------------------
st.title("🛡️ SSTBot")
st.subheader(
    "Asistente especializado en el Manual del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST)"
)

st.markdown("---")

pregunta = st.text_input(
    "Escribe tu pregunta:"
)

# -----------------------------
# BOTÓN
# -----------------------------
if st.button("Consultar"):

    if pregunta:

        with st.spinner("Analizando documento..."):

            documentos = retriever.invoke(pregunta)

            contexto = "\n\n".join(
                [doc.page_content for doc in documentos]
            )

            paginas = sorted(
                list(
                    set(
                        [
                            doc.metadata["page"] + 1
                            for doc in documentos
                        ]
                    )
                )
            )

            prompt_final = prompt.format(
                contexto=contexto,
                pregunta=pregunta
            )

            respuesta = llm.invoke(prompt_final)

        st.markdown("### Respuesta")

        st.markdown(
            f"""
            <div class='respuesta'>
            {respuesta.content}
            <br><br>
            <b>Páginas consultadas:</b> {paginas}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Por favor ingrese una pregunta.")