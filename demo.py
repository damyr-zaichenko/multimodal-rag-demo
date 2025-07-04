from scrapers.issue_article_scrapper import IssueArticleScraper
from processors.document_processor import DocumentProcessor
import streamlit as st


st.set_page_config(
    page_title="Parsing demo",  
)

scrapper = IssueArticleScraper('https://www.deeplearning.ai/the-batch/issue-302/')
parsed = scrapper.parse()
st.subheader('Parsed Page')
st.json(parsed)

processor = DocumentProcessor()
processed = processor.chunk(parsed)

st.divider()
st.subheader('Chunked page')
st.json(processed)