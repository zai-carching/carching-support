import os
import json
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "whatsapp-history-1"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Function to retrieve relevant context from Pinecone
def retrieve_context(query):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    query_emb = response.data[0].embedding

    result = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)

    contexts = []
    for match in result.matches:
        meta = match.metadata
        contexts.append(f"{meta['sender']}: {meta['message']}")

    return "\n".join(contexts)

# Function to get chatbot response
def respond(message, chat_history):
    context = retrieve_context(message)

    system_prompt = (
        "You are a helpful assistant for carching. Use the following past conversation data on whatsapp "
        "to answer the user's question if relevant:\n\n" + context
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages
    )

    bot_reply = response.choices[0].message.content

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    return chat_history

# Streamlit UI
st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬")

st.title("💬 Customer Support Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    st.chat_message("user").markdown(prompt)

    # Get response
    st.session_state.chat_history = respond(prompt, st.session_state.chat_history)

    # Display assistant response
    st.chat_message("assistant").markdown(st.session_state.chat_history[-1]["content"])
