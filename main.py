from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import instruction_str, new_prompt, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import space_engine
import streamlit as st

load_dotenv()

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=space_engine,
        metadata=ToolMetadata(
            name="Space",
            description="This gives detailed information about Space, Blackholes, Universe, Galaxies, Terraforming, and its Mystery.",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613", temperature=0.5)
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
st.set_page_config(
    page_title="Space Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dive into the Cosmos: Your Personal Space Guide")

# Assuming your image is in the same directory as your Streamlit script
image_path = "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?q=80&w=2565&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Center the image using HTML/CSS
st.markdown(
    f'<div style="display: flex; justify-content: center;">'
    f'<img src="{image_path}" alt="A mesmerizing view of the universe" style="width: 360px;">'
    f'</div>',
    unsafe_allow_html=True
)

prompt = st.text_input("Enter the most mysterious question you have about the universe ('q' to quit):")

if prompt != "q" and st.button("Submit"):
    result = agent.query(prompt)
    st.write(result.response)
