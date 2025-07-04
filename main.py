from query_engine import QueryEngine
from answer_generator import AnswerGenerator
from storage.faiss_chunk_store import FaissChunkStore
from embedding.embedder import ChunkEmbedder
import streamlit as st

store = FaissChunkStore(dim=384)
embedder = ChunkEmbedder()
engine = QueryEngine(store, embedder)
generator = AnswerGenerator()

st.set_page_config(
    page_title="RAG system",  
)

st.markdown("""
    <style>
    .stButton>button {
        font-size: 20px;
        padding: 0.5em 3em;
        border-radius: 8px;
        margin-top: 24px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Query input and search
@st.fragment
def query_input_fragment():
    col1, col2 = st.columns([5, 2])
    query = col1.text_input("Enter your query")
    if col2.button("Run", key='run_pressed'):
        rewritten_query = generator.rewrite_query(query)
        st.session_state["rewritten_query"] = rewritten_query
        result = engine.query(rewritten_query)
        st.session_state["retrieval_result"] = result
        st.session_state["query_text"] = query
        st.rerun()

# Display results and cost estimate before generation
@st.fragment
def results_fragment():
    if "retrieval_result" not in st.session_state:
        return

    result = st.session_state["retrieval_result"]
    original_query = st.session_state.get("query_text", "")
    rewritten_query = st.session_state.get("rewritten_query", "")

    st.markdown(f"**Original query:** {original_query}")
    st.markdown(f"**Rewritten query (optimized for more effective retrieval):** _{rewritten_query}_")

    top_k_results = result["results"][:5]

    #st.markdown(f"**Entities:** `{', '.join(result['entities']) or 'None'}`")

    for i, (meta, score) in enumerate(top_k_results, 1):
        st.divider()
        st.markdown(f"**{i}.** [{meta['article_url']}]({meta['article_url']}) — Score: {score:.4f}")
        st.markdown(f"<div style='font-size:18px'>{meta['text'][:800]}...</div>", unsafe_allow_html=True)
        if meta['image_url']:
            st.text(' ')
            st.image(meta['image_url'])

    # Show cost estimate before answer generation
    prompt = generator.build_prompt(rewritten_query, top_k_results)

    st.text(' ')
    with st.expander('Full prompt', expanded=False):
        st.code(prompt)

    st.text(' ')
    #generator.show_cost_estimate(prompt, generator.max_tokens)

    if st.button("Generate Answer", key="generate_button"):
        with st.spinner("Generating answer..."):
            answer = generator.generate(original_query, top_k_results)
            st.success("✅ Answer")
            with st.container(border=True):
                st.text(answer)
            

# 🚀 Main wrapper
def main():
    st.title("🔍 Multimodal RAG Assistant")
    query_input_fragment()
    results_fragment()


if __name__ == '__main__':
    main()