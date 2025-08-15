import os
import json
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# ✅ Must be first Streamlit call
st.set_page_config(page_title="Carching Support", page_icon="🚗")

# Load environment variables
load_dotenv(verbose=True)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "web-data"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 3  # Reduced for better relevance

# Initialize clients
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize services: {str(e)}")
    st.stop()

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Editable system prompt
default_prompt = """You are a helpful customer support assistant for Carching. 
Use the following information to answer the user's question. If you don't know the answer, say you'll find out.

If you're being asked in English, you can continue normally but if you're being asked in Bahasa Malaysia or Malay or even broken shortform of the malay language, you must sound like a mass market colloquial Malay that have the flexibility to speak in short forms as well. in short explain in style of borak warung.

Relevant Information:
{context}"""
system_prompt_template = st.sidebar.text_area("📝 System Prompt", value=default_prompt, height=250)

# Model selector
model_choice = st.sidebar.selectbox(
    "🤖 Select Model",
    ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo","gpt-5-mini"]
)

# Function to retrieve relevant context from Pinecone
def retrieve_context(query):
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        query_emb = response.data[0].embedding

        result = index.query(
            vector=query_emb,
            top_k=TOP_K,
            include_metadata=True
        )

        contexts = []
        for match in result.matches:
            meta = match.metadata
            context_str = f"Title: {meta['title']}\nContent: {meta['content']}"
            contexts.append(context_str)

        return "\n\n".join(contexts)
    except Exception as e:
        st.warning(f"Couldn't retrieve context: {str(e)}")
        return ""

# Function to get chatbot response
def respond(message, chat_history):
    try:
        context = retrieve_context(message)
        system_prompt = system_prompt_template.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "user", "content": message}
        ]

        response = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content

        updated_history = chat_history[-4:] + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_reply}
        ]

        return updated_history
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Sorry, I encountered an error. Please try again."}
        ]

# App Title
st.title("🚗 Carching Customer Support")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about Carching..."):
    st.chat_message("user").markdown(prompt)

    with st.spinner("Thinking..."):
        st.session_state.chat_history = respond(
            prompt,
            st.session_state.chat_history
        )

    st.chat_message("assistant").markdown(st.session_state.chat_history[-1]["content"])
