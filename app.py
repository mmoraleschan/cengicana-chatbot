import streamlit as st
import openai
import chromadb
from chromadb.utils import embedding_functions
import os

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-3-large"
)
chroma_client = chromadb.Client()
collection = chroma_client.get_collection(name="cengicana_docs", embedding_function=embedding_function)

st.title("🧠 Asistente Técnico de CENGICAÑA")
st.markdown("Consulta sobre productividad, variedades y zafras.")

pregunta = st.text_input("🔍 Escribe tu consulta")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        resultados = collection.query(query_texts=[pregunta], n_results=3)
        contexto = "\n\n".join(resultados["documents"][0])

        respuesta = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente técnico experto en caña de azúcar. Solo responde usando el contexto proporcionado."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
            ]
        )

        st.success("Respuesta del asistente:")
        st.write(respuesta.choices[0].message.content)

        with st.expander("🔎 Ver evidencia utilizada"):
            for i, frag in enumerate(resultados["documents"][0]):
                st.markdown(f"**Fragmento {i+1}:**\n{frag}")
