import os
import certifi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

os.environ['SSL_CERT_FILE'] = certifi.where()

class DatabaseManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.graph = Neo4jGraph(
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database,
            refresh_schema=False
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def wipe_database(self):
        """Clears all old nodes and relationships to prevent data leakage."""
        self.graph.query("MATCH (n) DETACH DELETE n;")

    def check_is_initialized(self):
        """Checks if nodes already exist in Neo4j."""
        try:
            res = self.graph.query("MATCH (n) RETURN count(n) as cnt")
            return res[0]['cnt'] > 0
        except Exception:
            return False

    def ingest_pdf(self, pdf_path):
        """Loads PDF, wipes database, creates Vector Store + Knowledge Graph."""
        self.wipe_database()
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = text_splitter.split_documents(docs)
        
        # 1. Store Text Vector Index
        Neo4jVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database,
            index_name="pdf_text_vector"
        )
        
        # 2. Extract & Store Knowledge Graph with explicit node and relation types
        llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Organization", "Company", "Franchise", "Person", "Coach", "Award", "Grant"
            ],
            allowed_relationships=[
                "OWNS", "CONTROLS", "FUNDS", "SPONSORS", "FOUNDED", "PARENT_COMPANY_OF", 
                "HEAD_COACH_OF", "PRESENTED_BY", "AWARDED_TO", "ASSOCIATED_WITH", "CHAIRPERSON_OF"
            ]
        )
        
        # Extract from whole pages: 300-char chunks split table rows from their
        # headers, which produced mis-attributed edges (e.g. assistant coach as head coach).
        graph_docs = llm_transformer.convert_to_graph_documents(docs)
        self._merge_duplicate_entities(graph_docs)

        try:
            self.graph.query(
                "CREATE CONSTRAINT unique_node_id IF NOT EXISTS FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
            )
        except Exception:
            pass
            
        self.graph.add_graph_documents(graph_docs, baseEntityLabel=True)
        return len(chunks), len(graph_docs)

    @staticmethod
    def _merge_duplicate_entities(graph_docs):
        """Collapses variants like 'Metro Warriors Franchise' onto 'Metro Warriors' in place."""
        generic_suffixes = (" franchise", " team", " club")
        ids = {n.id for gd in graph_docs for n in gd.nodes}
        lower_to_id = {i.lower(): i for i in ids}

        canonical = {}
        for node_id in ids:
            lowered = node_id.lower()
            for suffix in generic_suffixes:
                if lowered.endswith(suffix) and lowered[: -len(suffix)] in lower_to_id:
                    canonical[node_id] = lower_to_id[lowered[: -len(suffix)]]
                    break

        for gd in graph_docs:
            for node in gd.nodes:
                node.id = canonical.get(node.id, node.id)
            for rel in gd.relationships:
                rel.source.id = canonical.get(rel.source.id, rel.source.id)
                rel.target.id = canonical.get(rel.target.id, rel.target.id)
            # Drop nodes duplicated by the merge within the same document
            unique = {}
            for node in gd.nodes:
                unique.setdefault(node.id, node)
            gd.nodes = list(unique.values())

    def get_all_graph_edges(self):
        """Retrieves graph edges for visual rendering."""
        query = "MATCH (s)-[r]->(t) WHERE NOT s:Chunk RETURN s.id AS source, type(r) AS rel, t.id AS target LIMIT 60"
        return self.graph.query(query)

    def get_all_vector_chunks(self):
        """Retrieves all vector-indexed text chunks from Neo4j for inspection."""
        query = """
        MATCH (n:`_Neo4jVectorStore`) 
        RETURN n.text AS chunk_text, n.id AS chunk_id 
        LIMIT 50
        """
        try:
            results = self.graph.query(query)
            if results:
                return results
        except Exception:
            pass
        
        fallback_query = "MATCH (n) WHERE n.text IS NOT NULL RETURN n.text AS chunk_text, elementId(n) AS chunk_id LIMIT 50"
        return self.graph.query(fallback_query)

    def get_vector_store(self):
        return Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database,
            index_name="pdf_text_vector"
        )