"""
TRUE RAG APP - Ask ANY question about your cell analysis data

This is REAL RAG:
- Embeds ALL your data into vectors
- Retrieves relevant information for ANY question you ask
- Generates answers using LLM with retrieved context

No hardcoding. No anticipated queries. Just semantic search + generation.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page config
st.set_page_config(
    page_title="Cell Analysis RAG",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1a73e8;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .source-header {
        font-weight: bold;
        color: #1a73e8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load and initialize the RAG system (cached)."""
    try:
        from true_rag import TrueRAG
        rag = TrueRAG()
        return rag, None
    except ImportError as e:
        return None, f"Missing dependencies: {e}\n\nRun: pip install chromadb sentence-transformers"
    except Exception as e:
        return None, str(e)


def main():
    # Header
    st.markdown('<p class="main-header">Cell Analysis RAG</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask <b>ANY</b> question about your pancreatic cancer cell analysis data. '
        'True RAG - semantic search + LLM generation.</p>',
        unsafe_allow_html=True
    )

    # Load RAG system
    rag, error = load_rag_system()

    if error:
        st.error(f"Failed to load RAG system: {error}")
        st.info("Install required packages:\n```\npip install chromadb sentence-transformers\n```")
        return

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        n_results = st.slider(
            "Number of sources to retrieve",
            min_value=5,
            max_value=30,
            value=10,
            help="More sources = more context for the LLM"
        )

        if st.button("Reindex Data", type="secondary"):
            with st.spinner("Reindexing all data..."):
                try:
                    rag.index_data(force=True)
                    st.success("Reindexing complete!")
                except Exception as e:
                    st.error(f"Reindexing failed: {e}")

        st.divider()
        st.markdown("### How it works")
        st.markdown("""
        1. **Embed**: All data is converted to vectors
        2. **Retrieve**: Your question finds relevant data
        3. **Generate**: LLM answers using retrieved context

        This is TRUE RAG - no hardcoded queries!
        """)

        st.divider()
        st.markdown("### Example questions")
        examples = [
            "What happens to CD8 T cells in obesity?",
            "Which genes are in the glycolysis signature?",
            "How do macrophages change with BMI?",
            "What cell-cell interactions involve exhausted T cells?",
            "Compare inflammatory response across BMI categories",
            "What signatures increase in acinar cells?",
            "Which cells show metabolic changes in obese patients?",
            "Tell me about PD-L1 expression",
            "What is the oxidative phosphorylation signature?",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.question = ex

    # Initialize session state
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    # Check if data is indexed
    try:
        count = rag.collection.count() if rag.collection else 0
    except:
        count = 0

    if count == 0:
        st.warning("Data not yet indexed. Click 'Reindex Data' in the sidebar to start.")
        if st.button("Index Data Now", type="primary"):
            with st.spinner("Indexing all data (this may take a minute)..."):
                try:
                    rag.index_data(force=True)
                    st.success("Indexing complete! You can now ask questions.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        return

    st.success(f"Ready! {count:,} documents indexed.")

    # Question input
    question = st.text_input(
        "Ask any question about your data:",
        value=st.session_state.question,
        placeholder="e.g., What happens to T cells in obese patients?",
        key="question_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    # Process question
    if ask_button and question:
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = rag.query(question, n_results=n_results)

                # Store in history
                st.session_state.history.insert(0, result)

                # Clear the session state question
                st.session_state.question = ""

            except Exception as e:
                st.error(f"Error: {e}")
                return

    # Display results
    if st.session_state.history:
        for i, result in enumerate(st.session_state.history):
            if i > 0:
                st.divider()

            # Question
            st.markdown(f"**Q: {result['question']}**")

            # Answer
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(result['answer'])
            st.markdown('</div>', unsafe_allow_html=True)

            # Sources (collapsible)
            with st.expander(f"View {result['num_sources']} sources"):
                for j, source in enumerate(result['sources']):
                    st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                    st.markdown(f"<span class='source-header'>Source {j+1}</span> ({source['metadata'].get('source', 'unknown')})")
                    st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                    st.markdown(f"*Distance: {source['distance']:.4f}*")
                    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.divider()
    st.caption(
        "This is TRUE RAG - Retrieval Augmented Generation. "
        "Your question is embedded, relevant data is retrieved via semantic search, "
        "and an LLM generates the answer based on that context. No hardcoded queries."
    )


if __name__ == "__main__":
    main()
