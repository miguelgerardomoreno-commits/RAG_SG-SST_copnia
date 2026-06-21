from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

# Cargar variables de entorno
load_dotenv(dotenv_path=".env")

print(os.getenv("GROQ_API_KEY"))
# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cargar BD vectorial
vector_db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)

# Modelo Groq
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Prompt
prompt = PromptTemplate(
    input_variables=["contexto", "pregunta"],
    template="""
Eres "SSTBot", asistente especializado en el Manual del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST) del COPNIA.

Tu rol es responder preguntas únicamente utilizando la información del contexto proporcionado.

Reglas obligatorias:

1. Responde únicamente con la información del contexto.
2. No inventes información.
3. Si la respuesta no se encuentra en el contexto responde exactamente:

"No encontré información sobre esa pregunta en el manual SG-SST."

4. Redacta respuestas claras y profesionales.
5. Siempre cita al final las páginas consultadas.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""
)

print("=" * 60)
print("SSTBot - Asistente SG-SST")
print("Escribe 'salir' para finalizar")
print("=" * 60)

while True:

    pregunta = input("\nPregunta: ")

    if pregunta.lower() == "salir":
        print("Hasta pronto.")
        break

    documentos = retriever.invoke(pregunta)

    contexto = "\n\n".join(
        [doc.page_content for doc in documentos]
    )

    paginas = sorted(
        list(
            set(
                [doc.metadata["page"] + 1 for doc in documentos]
            )
        )
    )

    prompt_final = prompt.format(
        contexto=contexto,
        pregunta=pregunta
    )

    respuesta = llm.invoke(prompt_final)

    print("\nRespuesta:\n")
    print(respuesta.content)

    print(f"\nPáginas consultadas: {paginas}")