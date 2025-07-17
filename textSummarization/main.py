import validators
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
import traceback
from langchain.schema import Document




load_dotenv()
groq_key = os.getenv("GROQ_KEY")
google_key=os.getenv("G_KEY")

llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_key)

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

generic_url = st.text_input("URL", label_visibility="collapsed")

## LLM


prompt_template = """
Provide a summary of the following content in complete accurate:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_youtube_transcript_text(youtube_url):
    parsed = urlparse(youtube_url)
    qs = parse_qs(parsed.query)
    video_id = qs.get('v', [''])[0]
    if not video_id:
        raise ValueError("Invalid YouTube URL — missing video ID")
    
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry["text"] for entry in transcript_list])
    return text

if st.button("Summarize the Content from YT or Website"):
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    text = get_youtube_transcript_text(generic_url)
                    docs = [Document(page_content=text)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.error(f"Exception: {str(e)}")
            st.text("Stack trace:")
            st.text(traceback.format_exc())
