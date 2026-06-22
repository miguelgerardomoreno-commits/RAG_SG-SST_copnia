# Sistema RAG sobre el Manual SG-SST del COPNIA

## Información General

**Estudiante:** Miguel Gerardo Moreno
**Fecha:** 20 de junio de 2026

---

## Descripción del Proyecto

Este proyecto consiste en el desarrollo de un sistema de Recuperación Aumentada por Generación (Retrieval-Augmented Generation, RAG) capaz de responder preguntas en lenguaje natural a partir del contenido del Manual del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST) del COPNIA.

El sistema permite consultar información especializada contenida en el documento y generar respuestas fundamentadas utilizando técnicas de procesamiento de lenguaje natural, embeddings y bases de datos vectoriales.

---

## Documento Seleccionado y Justificación

### Documento seleccionado

**Manual del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST) del COPNIA.**

### Justificación

Se seleccionó este documento debido a su carácter técnico y normativo, además de la relevancia que tiene la Seguridad y Salud en el Trabajo dentro de las organizaciones. El manual contiene procedimientos, responsabilidades, políticas y lineamientos específicos que lo convierten en un candidato ideal para implementar un sistema RAG, permitiendo la consulta eficiente de información especializada.

---

## Persona Usuaria Objetivo y Caso de Uso

### Persona usuaria objetivo

Funcionarios administrativos, profesionales en Seguridad y Salud en el Trabajo, estudiantes y personal interesado en consultar información relacionada con el SG-SST.

### Caso de uso

El sistema permite que los usuarios realicen preguntas en lenguaje natural sobre el Manual SG-SST del COPNIA y obtengan respuestas precisas basadas únicamente en el contenido del documento, evitando la necesidad de revisar manualmente el PDF completo.

---

## Tecnologías Utilizadas

* Python
* LangChain
* Groq
* HuggingFace Embeddings
* ChromaDB
* Streamlit
* Sentence Transformers

---

## Arquitectura del Sistema

1. Carga del documento PDF.
2. División del documento en fragmentos (chunks).
3. Generación de embeddings utilizando HuggingFace.
4. Almacenamiento de embeddings en ChromaDB.
5. Recuperación de información relevante mediante búsqueda semántica.
6. Generación de respuestas utilizando el modelo Llama 3 a través de Groq.
7. Presentación de resultados mediante una interfaz desarrollada en Streamlit.

---

## Preguntas y Respuestas Generadas por el Sistema

### Pregunta 1

**¿Cuál es la política del SG-SST?**

**Respuesta:**

🛡️ No encontré información sobre la política del SG-SST en el contexto proporcionado. Sin embargo, puedo proporcionar información sobre la política de Seguridad y Salud en el Trabajo mencionada en el historial:

La política de Seguridad y Salud en el Trabajo debe ser aprobada por el Director General a través de un documento escrito, y debe proporcionar un marco de referencia para establecer y revisar los objetivos de Seguridad y Salud en el Trabajo.

Si se busca información sobre la política del SG-SST en general, no hay una sección específica que la defina en el contexto proporcionado.
📄 Páginas consultadas: [6, 17, 18, 19, 21, 26, 27]

---

### Pregunta 2

**¿Cuáles son las responsabilidades del Director General?**

**Respuesta:**

🛡️ Las responsabilidades del Director General son:

• Máximo responsable del Sistema de Gestión de Seguridad y Salud en el Trabajo • Aprobar la política de Seguridad y Salud en el Trabajo a través de documento escrito, la cual deberá proporcionar un marco de referencia para establecer y revisar los objetivos de Seguridad y Salud en el Trabajo • Informar a la Dirección General sobre las actividades y situaciones que se presenten con la implementación del Sistema de Gestión de Seguridad y Salud en el Trabajo • Promover la cultura de autocuidado.
📄 Páginas consultadas: [1, 13, 16, 17, 18, 19, 20]
---

### Pregunta 3

**¿Qué funciones tiene el COPASST?**

**Respuesta:**

🛡️ El Comité Paritario de Seguridad y Salud en el Trabajo (COPASST) es un organismo de promoción y vigilancia de las normas y reglamentos de Seguridad y Salud en el Trabajo dentro de la empresa. No se ocupará de tramitar asuntos referentes a la relación contractual - laboral propiamente dicha, los problemas de personal, disciplinario o sindicales.

Sus funciones no están explícitamente detalladas en el contexto proporcionado, pero se menciona que es un organismo de promoción y vigilancia de las normas y reglamentos de Seguridad y Salud en el Trabajo.
📄 Páginas consultadas: [2, 3, 8, 9, 11, 13, 21, 25, 26, 27, 28]

---

### Pregunta 4

**¿Qué establece el programa de promoción y prevención?**

**Respuesta:**

🛡️ El programa de promoción y prevención en la salud busca incentivar los estilos de vida saludable, la prevención del consumo de sustancias psicoactivas y la salud pública. Algunos de los temas establecidos para desarrollar son:

• Conservación Visual
• Conservación Auditiva
• Cuidado Vocal
• Riesgo Cardiovascular
• Salud Pública
📄 Páginas consultadas: [5, 6, 8, 10, 12, 16, 18, 23, 24]

---

### Pregunta 5

**¿Cómo se reportan los incidentes o accidentes de trabajo?**

**Respuesta:**

🛡️ Se cuenta con el procedimiento (GH-pr-11) Reporte e investigación de incidentes y accidentes de trabajo, el cual busca establecer el paso a paso para reportar, investigar y analizar las causas, hechos y situaciones que han generado incidentes o accidentes de trabajo, con el fin de tomar medidas correctivas, preventivas y de mejoramiento, que permiten intervenir las causas y se establece que se deben:

• Adelantar las investigaciones de los incidentes y accidentes de trabajo dentro del plazo establecido en el numeral 2 de la Resolución 1401 de 2007 y presentar informes al responsable del sistema de seguridad y salud en el trabajo, de acuerdo con la metodología y el formato que adopte el COPNIA para tal fin.
📄 Páginas consultadas: [7, 10, 12, 19, 22, 26]

---

## Instrucciones para Ejecutar el Sistema

### 1. Clonar el repositorio

```bash
git clone https://github.com/miguelgerardomoreno-commits/RAG_SG-SST_copnia.git
cd RAG_SG-SST_copnia
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv venv
```

Activar en Windows:

```bash
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API Key de Groq

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
GROQ_API_KEY=tu_api_key
```

### 5. Generar la base vectorial

```bash
python crear_embeddings.py
```

### 6. Ejecutar la aplicación

```bash
streamlit run app.py
```

---

## Estructura del Proyecto

```text
RAG_SG-SST_copnia/
│
├── app.py
├── cargar_documento.py
├── crear_embeddings.py
├── chat_rag.py
├── manual_SG-SST.pdf
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

---

## Video de Demostración

Link del video:

[Agregar enlace al video de demostración]

---

## Conclusiones

Durante el desarrollo del proyecto se aprendió el funcionamiento de los sistemas RAG, integrando técnicas de recuperación semántica, embeddings y modelos de lenguaje para responder preguntas sobre documentos especializados.

Como limitación, se identificó que la calidad de las respuestas depende directamente de la calidad de los fragmentos recuperados y de la estructura del documento utilizado.
