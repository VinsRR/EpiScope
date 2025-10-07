"""
ui/streamlit_app.py

A simple Streamlit user interface for EpiScope.

This app allows users to ingest PDFs into the system and pose queries.
Answers are displayed along with citations (evidence snippets).
"""
import streamlit as st
from typing import Optional

from ..ingest.pipeline import IngestPipeline
from ..generate.answer import AnswerGenerator

st.set_page_config(page_title="EpiScope", layout="wide")
st.title("EpiScope – Evidence‑based Q&A for Papers")

# Sidebar controls
with st.sidebar:
    st.header("Ingest Documents")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    ingest_status: Optional[str] = None
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pipeline = IngestPipeline(rag_method="text")
        pipeline.ingest_pdf(temp_path)
        ingest_status = f"Indexed {uploaded_file.name}"
        st.success(ingest_status)
    doi = st.text_input("Ingest by DOI")
    if doi:
        pipeline = IngestPipeline(rag_method="text")
        pipeline.ingest_doi(doi)
        st.success(f"Requested ingestion for DOI {doi}")

# Main query area
st.header("Ask a Question")
query = st.text_input("Question")
top_k = st.slider("Number of contexts", min_value=1, max_value=10, value=5)
if st.button("Get Answer") and query:
    generator = AnswerGenerator(rag_method="text")
    prov = generator.answer_question(query, top_k=top_k)
    st.subheader("Answer")
    st.write(prov.answer)
    st.subheader("Evidence")
    for ev in prov.evidences:
        with st.expander(f"Paper {ev.paper_id} – Section {ev.section}"):
            st.write(ev.snippet)
