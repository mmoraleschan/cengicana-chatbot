
import streamlit as st
import openai
import os
import json
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configurar clave API
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Cargar fragmentos desde archivo
with open("fragmentos.json", "r", encoding="utf-8") as f:
    fragmentos = json.load(f)

# Obtener embeddings para todos los fragmentos al inicio
@st.cache_resource
def generar_embeddings_fragmentos():
    textos = [frag["text"] for frag in fragmentos]
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=textos
    )
    embeddings = [item.embedding for item in response.data]
    return textos, embeddings

fragmentos_texto, fragmentos_embeddings = generar_embeddings_fragmentos()

# Procesar la pregunta y buscar los fragmentos más similares
def responder_pregunta(pregunta: str):
    pregunta_embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=[pregunta]
    ).data[0].embedding

    sims = cosine_similarity([pregunta_embedding], fragmentos_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:3]
    top_fragmentos = [fragmentos_texto[i] for i in top_indices]

    contexto = "\n\n".join(top_fragmentos)
    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente técnico experto en caña de azúcar. Solo responde usando el contexto proporcionado."},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
        ]
    )
    return respuesta.choices[0].message.content, top_fragmentos

# Interfaz Streamlit
st.title("🧠 Asistente Técnico de CENGICAÑA (Prototipo Web)")
st.markdown("Consulta sobre productividad, variedades y zafras. Prototipo sin base de datos externa.")

pregunta = st.text_input("🔍 Escribe tu consulta")

if pregunta:
    with st.spinner("Generando respuesta..."):
        respuesta, evidencia = responder_pregunta(pregunta)
        st.success("✅ Respuesta del asistente:")
        st.write(respuesta)

        with st.expander("📄 Ver fragmentos usados"):
            for i, frag in enumerate(evidencia):
                st.markdown(f"**Fragmento {i+1}:**\n{frag}")
