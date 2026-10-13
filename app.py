import os
import re
import streamlit as st
import graphviz
from dotenv import load_dotenv
from src.db_manager import DatabaseManager
from src.rag_engine import RAGEngine

load_dotenv()

st.set_page_config(page_title="GraphRAG vs. Vector RAG Enterprise Benchmark", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-header { font-size: 14px; color: #64748B; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Hybrid GraphRAG vs. Vector RAG Enterprise Benchmark</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Evaluating multi-hop reasoning, tabular relationships, and exact numerical precision against isolated retrieval architectures.</p>', unsafe_allow_html=True)

@st.cache_resource
def init_pipeline():
    db = DatabaseManager()
    engine = RAGEngine(db)
    return db, engine

db, engine = init_pipeline()

# Ensure this matches your exact filename in assets/
pdf_path = "assets/2026_inter_city_cricket_championship_enterprise_report.pdf"

# Auto-ingest if database is empty
if not db.check_is_initialized():
    if os.path.exists(pdf_path):
        with st.spinner("Initializing Neo4j database and auto-ingesting structured benchmark PDF..."):
            db.ingest_pdf(pdf_path)

# --- SIDEBAR ---
st.sidebar.header("Configuration")
if st.sidebar.button("🔄 Force Re-Ingest PDF", width="stretch"):
    with st.spinner("Wiping database and re-indexing structured document..."):
        chunks, graph_docs = db.ingest_pdf(pdf_path)
        st.session_state.pop("eval_results", None)
        st.sidebar.success(f"Indexed {chunks} chunks & {graph_docs} graph docs.")

st.sidebar.markdown("---")
st.sidebar.info("**Benchmark Architecture:**\n- **Vector RAG:** Semantic similarity search over partitioned text chunks.\n- **Graph RAG:** Multi-hop structural graph traversal.\n- **Hybrid Search:** Combined graph traversal + text chunks.")

# Curated benchmark questions matching document partitions (Pages 1, 2, 3)
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the Head Coach's name of the team owned by Apex Sports Group?",
        "type": "Cross-page multi-hop test (Page 2 Ownership -> Page 3 Coaching)",
        "expected_vector": "Partial / Failed (Chunks are partitioned across pages)",
        "expected_graph": "Success (Traverses Apex -> Royal Strikers -> Ramesh Kumar)",
        "expected_answer": "Ramesh Kumar",
        "expected_patterns": [r"ramesh kumar"]
    },
    {
        "id": "Q2",
        "question": "What was Arjun Mehra's exact strike rate and how many sixes did he hit?",
        "type": "Exact numerical/float precision test (Page 1 Table)",
        "expected_vector": "Success (Direct tabular chunk match)",
        "expected_graph": "Failed (Floats/numeric metrics often dropped during graph extraction)",
        "expected_answer": "Strike rate 203.63, 8 sixes",
        "expected_patterns": [r"203\.63", r"\b(8|eight)\b"]
    },
    {
        "id": "Q3",
        "question": "Which organization is the primary financial backer of the institute that founded the Metro Warriors?",
        "type": "Deep multi-hop corporate hierarchy test (Page 2)",
        "expected_vector": "Partial (Requires connecting multiple nested ownership clauses)",
        "expected_graph": "Success (Traverses Meridian -> Zurich -> Metro Warriors)",
        "expected_answer": "Meridian Corp",
        "expected_patterns": [r"meridian"]
    },
    {
        "id": "Q4",
        "question": "Who presented the trophy to Arjun Mehra, and what was the Meridian Corp grant allocation?",
        "type": "Cross-entity & cross-page financial linkage test (Page 2 & Page 3)",
        "expected_vector": "Partial (Information split between financial register and award log)",
        "expected_graph": "Success (Links award recipient to Tournament Director & grant)",
        "expected_answer": "Mr. Sen Gupta; $50,000",
        "expected_patterns": [r"sen gupta", r"50,?000"]
    }
]

# --- MAIN TABS ---
tab_comp, tab_graph, tab_vector, tab_pdf = st.tabs([
    "📊 Evaluation Matrix", 
    "🕸️ Knowledge Graph Visualizer", 
    "🗂️ Vector Store Chunks", 
    "📄 Document Viewer"
])

with tab_pdf:
    st.subheader("Source Benchmark Document Viewer")
    if os.path.exists(pdf_path):
        st.markdown("""
        **Document Design:** Structured with explicit partitions:
        * **Page 1:** Match Summary & Player Statistics Table.
        * **Page 2:** Corporate Ownership & Holding Structure Table.
        * **Page 3:** Coaching Directory & Official Presentation Log.
        """)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.pdf(pdf_bytes, height=900)
        st.download_button("Download Benchmark PDF", pdf_bytes, file_name="2026_inter_city_cricket_championship_enterprise_report.pdf")
    else:
        st.error(f"Missing PDF at `{pdf_path}`. Please verify your filename inside `assets/`.")

with tab_graph:
    st.subheader("Neo4j Knowledge Graph Visualizer")
    edges = db.get_all_graph_edges()
    if edges:
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR', size='8,4')
        for e in edges:
            dot.node(e['source'], shape='ellipse', style='filled', color='#E0F2FE')
            dot.node(e['target'], shape='ellipse', style='filled', color='#DCFCE7')
            dot.edge(e['source'], e['target'], label=e['rel'])
        st.graphviz_chart(dot, width="stretch")
    else:
        st.info("No graph data available. Click Force Re-Ingest in the sidebar.")

with tab_vector:
    st.subheader("Vector Store Indexed Chunks")
    chunks = db.get_all_vector_chunks()
    if chunks:
        st.write(f"Total Chunks Indexed: {len(chunks)}")
        for i, c in enumerate(chunks):
            with st.expander(f"Chunk {i+1}"):
                st.write(c.get('chunk_text'))
    else:
        st.info("No vector chunks available.")

def evaluate_status(answer, patterns):
    lower_ans = answer.lower()
    hits = sum(1 for p in patterns if re.search(p, lower_ans))
    if hits == len(patterns):
        return "✅ **Success** (All expected facts present)"
    elif hits > 0:
        return f"⚠️ **Partial** ({hits}/{len(patterns)} expected facts present)"
    else:
        return "❌ **Failed** (Expected facts missing from answer)"

MODES = [
    ("vector", "Vector Search Only", "🔍 Debug: Retrieved Vector Chunks"),
    ("graph", "Knowledge Graph Only", "🔍 Debug: Retrieved Graph Paths"),
    ("hybrid", "Hybrid GraphRAG", "🔍 Debug: Combined Context"),
]

with tab_comp:
    st.markdown("### Architectural Benchmark Matrix")
    st.markdown("Comparing retrieval performance across partitioned tables and multi-hop relationships:")

    if st.button("▶️ Run Evaluation", type="primary"):
        results = {}
        progress = st.progress(0.0, text="Starting evaluation...")
        for i, item in enumerate(EVAL_QUESTIONS):
            progress.progress(i / len(EVAL_QUESTIONS), text=f"Evaluating {item['id']}...")
            results[item['id']] = {mode: engine.generate_answer(item['question'], mode) for mode, _, _ in MODES}
        progress.empty()
        st.session_state["eval_results"] = results

    results = st.session_state.get("eval_results")
    if not results:
        st.info("Click **Run Evaluation** to query all benchmark questions across Vector, Graph and Hybrid retrieval.")

    for item in EVAL_QUESTIONS:
        st.markdown(f"#### {item['id']}: {item['question']}")
        st.caption(f"Category: *{item['type']}* | Expected answer: **{item['expected_answer']}** | Design Expectation — Vector: `{item['expected_vector']}` | Graph: `{item['expected_graph']}`")

        if results and item['id'] in results:
            for col, (mode, title, debug_label) in zip(st.columns(3), MODES):
                answer, context = results[item['id']][mode]
                with col:
                    st.markdown(f"**{title}**")
                    st.markdown(evaluate_status(answer, item['expected_patterns']))
                    st.write(answer)
                    with st.expander(debug_label):
                        st.code(context, language="text")

        st.markdown("---")
