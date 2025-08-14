import os
import json
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=True)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "web-data"  # Updated index name
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"  # Updated to valid model
TOP_K = 3  # Reduced for better relevance

# Initialize clients
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize services: {str(e)}")
    st.stop()


# Function to retrieve relevant context from Pinecone
def retrieve_context(query):
    try:
        # Get embedding for the query
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        query_emb = response.data[0].embedding

        # Query Pinecone
        result = index.query(
            vector=query_emb,
            top_k=TOP_K,
            include_metadata=True
        )

        # Format context from matches
        contexts = []
        for match in result.matches:
            meta = match.metadata
            # Updated to match your website/FAQ data structure
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

        system_prompt = """You are a helpful customer support assistant for Carching. 
        Use the following information to answer the user's question. If you don't know the answer, say you'll find out.

        Relevant Information:
        {context}""".format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "user", "content": message}
        ]

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content

        # Update chat history (keeping last 6 exchanges)
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


# Streamlit UI
st.set_page_config(page_title="Carching Support", page_icon="🚗")

st.title("🚗 Carching Customer Support")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about Carching..."):
    # Display user message
    st.chat_message("user").markdown(prompt)

    # Get and display assistant response
    with st.spinner("Thinking..."):
        st.session_state.chat_history = respond(
            prompt,
            st.session_state.chat_history
        )

    st.chat_message("assistant").markdown(st.session_state.chat_history[-1]["content"])