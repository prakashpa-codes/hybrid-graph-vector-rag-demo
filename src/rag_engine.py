import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

STOPWORDS = {
    "the", "and", "who", "what", "which", "was", "were", "how", "many", "did", "does",
    "that", "this", "with", "for", "from", "his", "her", "its", "name", "exact",
}

class RAGEngine:
    def __init__(self, db_manager):
        self.db = db_manager
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def retrieve_vector(self, query):
        vector_store = self.db.get_vector_store()
        docs = vector_store.similarity_search(query, k=2)
        context = "\n\n".join([f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
        return context

    def retrieve_graph(self, query):
        keywords = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2 and w.lower() not in STOPWORDS]
        if not keywords:
            keywords = [w.lower() for w in re.findall(r'\b\w+\b', query)]

        # Anchor only on extracted entities (not vector Chunk nodes), and emit each
        # relationship as a directed triple so the LLM sees who -[REL]-> whom.
        cypher = """
        MATCH (s)
        WHERE s.id IS NOT NULL AND NOT s:Chunk
          AND ANY(kw IN $keywords WHERE toLower(s.id) CONTAINS kw)
        MATCH path = (s)-[*1..2]-(t)
        WITH path ORDER BY length(path)
        UNWIND relationships(path) AS r
        WITH DISTINCT startNode(r).id + ' -[' + type(r) + ']-> ' + endNode(r).id AS triple
        RETURN triple
        LIMIT 40
        """
        try:
            results = self.db.graph.query(cypher, {"keywords": keywords})
        except Exception:
            results = []

        if not results:
            results = self.db.graph.query(
                "MATCH (s)-[r]->(t) RETURN s.id + ' -[' + type(r) + ']-> ' + t.id AS triple LIMIT 25"
            )

        return "\n".join(r["triple"] for r in results)

    def generate_answer(self, query, mode):
        if mode == "vector":
            context = self.retrieve_vector(query)
            sys_msg = "You are a Vector RAG assistant. Answer using ONLY the provided text chunk context. If information is missing from the chunks, state that you cannot find it."
        elif mode == "graph":
            context = self.retrieve_graph(query)
            sys_msg = "You are a Graph RAG assistant. Answer using ONLY the provided Knowledge Graph relationship context. Each line is a directed triple 'Source -[RELATION]-> Target'; chain triples to answer multi-hop questions. If relations or exact numerical values are missing from graph properties, state that you cannot find them."
        else:  # hybrid
            v_ctx = self.retrieve_vector(query)
            g_ctx = self.retrieve_graph(query)
            context = f"--- VECTOR CHUNKS ---\n{v_ctx}\n\n--- KNOWLEDGE GRAPH RELATIONS ---\n{g_ctx}"
            sys_msg = "You are a Hybrid RAG assistant. Combine BOTH text chunks and Knowledge Graph relationships (directed triples 'Source -[RELATION]-> Target') to form a complete, verified answer."

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        return answer, context