import os
import streamlit as st
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from firecrawl import FirecrawlApp
from elevenlabs.client import ElevenLabs

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast (LangGraph + Groq)")

st.sidebar.header("🔑 API Keys")
groq_key = st.sidebar.text_input("Groq API Key", type="password")
elevenlabs_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
firecrawl_key = st.sidebar.text_input("Firecrawl API Key", type="password")

url = st.text_input("Enter Blog URL")

# ---------------------------
# Graph State
# ---------------------------
class PodcastState(TypedDict):
    url: str
    blog_text: str
    summary: str
    audio_bytes: bytes

# ---------------------------
# Nodes
# ---------------------------
def scrape_blog(state: PodcastState):
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    result = app.scrape(state["url"], formats= ["markdown"])
    return {"blog_text": result.markdown}

def summarize_blog(state: PodcastState):
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.3
    )

    prompt = f"""
    Convert the following blog into a conversational podcast script.
    Max 2000 characters.
    Tone: engaging, clear, human-like.

    BLOG:
    {state["blog_text"]}
    """

    response = llm.invoke(prompt)
    return {"summary": response.content}

def generate_audio(state: PodcastState):
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    audio_stream = client.text_to_speech.convert(
        text=state["summary"],
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2"
    )

    chunks = []
    for chunk in audio_stream:
        if chunk:
            chunks.append(chunk)

    return {"audio_bytes": b"".join(chunks)}

# ---------------------------
# LangGraph Definition
# ---------------------------
builder = StateGraph(PodcastState)

builder.add_node("scrape", scrape_blog)
builder.add_node("summarize", summarize_blog)
builder.add_node("audio", generate_audio)

builder.set_entry_point("scrape")
builder.add_edge("scrape", "summarize")
builder.add_edge("summarize", "audio")
builder.add_edge("audio", END)

graph = builder.compile()

# ---------------------------
# Run
# ---------------------------
if st.button("🎙️ Generate Podcast", disabled=not all([groq_key, elevenlabs_key, firecrawl_key])):
    if not url.strip():
        st.warning("Please enter a blog URL")
    else:
        try:
            with st.spinner("Processing with LangGraph agents..."):
                os.environ["GROQ_API_KEY"] = groq_key
                os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
                os.environ["FIRECRAWL_API_KEY"] = firecrawl_key

                final_state = graph.invoke({
                    "url": url,
                    "blog_text": "",
                    "summary": "",
                    "audio_bytes": b""
                })

            st.success("Podcast generated 🎧")
            st.audio(final_state["audio_bytes"], format="audio/mp3")

            st.download_button(
                "Download Podcast",
                final_state["audio_bytes"],
                "podcast.mp3",
                "audio/mp3"
            )

            with st.expander("📄 Podcast Script"):
                st.write(final_state["summary"])

        except Exception as e:
            st.error(f"Error: {e}")
